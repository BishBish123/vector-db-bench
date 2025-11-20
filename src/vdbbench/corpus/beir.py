"""BEIR-format corpus loader (MS-MARCO + every other dataset on the BeIR hub).

Why BEIR? Every published BEIR dataset ships in the same `corpus / queries /
qrels` shape with curated qrels, which matches `CorpusBundle` exactly. That
gives us reliable ground truth without having to fabricate synthetic
relevance judgements.

The loader has two layers:

* `bundle_from_beir_records()` — pure, in-memory adapter that turns three
  iterables into a `CorpusBundle`. Unit-testable with toy fixtures.
* `load_beir_dataset()` / `load_msmarco()` — IO wrappers that stream from
  the Hugging Face hub and feed the records through the pure layer. The
  streaming path is critical for MS-MARCO (8M passages) — see the sampling
  notes inside `load_beir_dataset`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from vdbbench.corpus.bundle import CorpusBundle


@dataclass(frozen=True)
class BeirRecord:
    """A single BEIR row, normalized to strings + a relevance grade."""

    pid: str
    text: str


@dataclass(frozen=True)
class BeirQuery:
    qid: str
    text: str


@dataclass(frozen=True)
class BeirQrel:
    qid: str
    pid: str
    relevance: float


def bundle_from_beir_records(
    name: str,
    corpus: Iterable[BeirRecord | dict[str, Any]],
    queries: Iterable[BeirQuery | dict[str, Any]],
    qrels: Iterable[BeirQrel | dict[str, Any]],
    metadata: dict[str, object] | None = None,
    drop_orphan_qrels: bool = True,
) -> CorpusBundle:
    """Build a `CorpusBundle` from BEIR-shaped iterables.

    Inputs may be `BeirRecord/Query/Qrel` instances *or* plain dicts whose
    keys match the dataclass fields (BEIR text fields like `title`/`text`
    get joined when both are present, matching the BEIR convention).

    `drop_orphan_qrels=True` quietly removes qrels referencing missing
    passages or queries — BEIR ships a small number of these and we'd rather
    not fail the loader, but the bundle's own validators *will* fail if they
    leak through, so the cleaning happens here at the boundary.
    """
    passage_rows: list[dict[str, str]] = []
    seen_pids: set[str] = set()
    for raw_p in corpus:
        rec = _coerce_record(raw_p)
        if rec.pid in seen_pids:
            continue  # silently de-dupe (BEIR has occasional repeats)
        seen_pids.add(rec.pid)
        passage_rows.append({"pid": rec.pid, "text": rec.text})

    query_rows: list[dict[str, str]] = []
    seen_qids: set[str] = set()
    for raw_q in queries:
        q = _coerce_query(raw_q)
        if q.qid in seen_qids:
            continue
        seen_qids.add(q.qid)
        query_rows.append({"qid": q.qid, "text": q.text})

    qrel_rows: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for raw_r in qrels:
        qr = _coerce_qrel(raw_r)
        key = (qr.qid, qr.pid)
        if key in seen_pairs:
            continue  # de-dupe before validators reject
        seen_pairs.add(key)
        qrel_rows.append({"qid": qr.qid, "pid": qr.pid, "relevance": qr.relevance})

    qrels_df = pd.DataFrame(qrel_rows, columns=["qid", "pid", "relevance"])
    if drop_orphan_qrels:
        qrels_df = qrels_df[qrels_df["pid"].isin(seen_pids) & qrels_df["qid"].isin(seen_qids)]
        qrels_df = qrels_df.reset_index(drop=True)

    return CorpusBundle(
        name=name,
        passages=pd.DataFrame(passage_rows, columns=["pid", "text"]),
        queries=pd.DataFrame(query_rows, columns=["qid", "text"]),
        qrels=qrels_df,
        metadata={"source": "beir", **(metadata or {})},
    )


# ---------------------------------------------------------------------------
# IO wrappers
# ---------------------------------------------------------------------------


def load_beir_dataset(
    dataset_name: str,
    split: str = "test",
    sample_size: int | None = None,
    seed: int = 42,
    cache_dir: str | Path | None = None,
) -> CorpusBundle:
    """Load a BEIR dataset from the HF hub and (optionally) sample passages.

    `dataset_name` is the BeIR repo suffix, e.g. `"msmarco"`, `"scifact"`,
    `"nfcorpus"`. `split` selects the qrels split (`"test"` for most BEIR
    datasets; MS-MARCO uses `"dev"`).

    Sampling strategy when `sample_size` is set:

    1. Read qrels fully (these are small — tens of thousands of rows even
       for MS-MARCO).
    2. Always include every judged passage in the sample. Without this,
       random sub-sampling of an 8M-row corpus down to 100k drops nearly
       every positive and silently invalidates recall numbers.
    3. Reservoir-sample the remaining unjudged passages to reach
       `sample_size`. Reservoir keeps memory bounded by `sample_size`
       instead of the full corpus.

    Memory bound: O(sample_size + |judged_pids|) regardless of corpus
    size, so MS-MARCO defaults are usable on a laptop.

    The download lives behind the `datasets` package — install with
    `uv sync --extra embed` or run inside the Docker image.
    """
    try:
        from datasets import load_dataset  # noqa: PLC0415  -- optional import
    except ImportError as e:  # pragma: no cover - env-dependent
        raise ImportError(
            "datasets is required for `load_beir_dataset`; install via "
            "`uv sync --extra embed` or use the Docker image."
        ) from e

    cache_path = str(cache_dir) if cache_dir else None
    repo = f"BeIR/{dataset_name}"

    qrels_ds = load_dataset(f"BeIR/{dataset_name}-qrels", split=split, cache_dir=cache_path)
    qrels_records: list[BeirQrel] = []
    judged_pids: set[str] = set()
    judged_qids: set[str] = set()
    for raw in qrels_ds:
        qr = _coerce_qrel(raw)
        qrels_records.append(qr)
        judged_pids.add(qr.pid)
        judged_qids.add(qr.qid)

    queries_ds = load_dataset(repo, "queries", split="queries", cache_dir=cache_path)
    queries_records = [
        q for q in (_coerce_query(raw) for raw in queries_ds) if q.qid in judged_qids
    ]

    if sample_size is None:
        corpus_records: list[BeirRecord] = [
            _coerce_record(raw)
            for raw in load_dataset(repo, "corpus", split="corpus", cache_dir=cache_path)
        ]
    else:
        if sample_size < len(judged_pids):
            raise ValueError(
                f"sample_size={sample_size} is smaller than the number of "
                f"judged passages in {repo}.{split} ({len(judged_pids)}). "
                f"Sub-sampling below this size would silently drop gold labels "
                f"and corrupt recall/NDCG. Bump sample_size or pick a smaller "
                f"BEIR dataset (e.g. scifact, nfcorpus)."
            )
        corpus_stream = load_dataset(
            repo, "corpus", split="corpus", cache_dir=cache_path, streaming=True
        )
        corpus_records = list(
            _stream_sample_corpus(
                stream=(_coerce_record(raw) for raw in corpus_stream),
                judged_pids=judged_pids,
                sample_size=sample_size,
                seed=seed,
            )
        )

    # Normalize no-op sampling so two `load_msmarco(sample_size=None)` calls
    # produce identical bundle identities regardless of `seed`.
    is_full = sample_size is None
    name_suffix = "all" if is_full else str(sample_size)
    seed_suffix = "0" if is_full else str(seed)
    bundle = bundle_from_beir_records(
        name=f"beir/{dataset_name}.{split}@n={name_suffix}.seed={seed_suffix}",
        corpus=corpus_records,
        queries=queries_records,
        qrels=qrels_records,
        metadata={
            "split": split,
            "hf_repo": repo,
            "sample_size": None if is_full else sample_size,
            "sample_seed": 0 if is_full else seed,
        },
    )

    # `judged_qids` was captured from the *raw* qrels; once orphan qrels are
    # filtered out by the bundle layer, some of those qids may have lost
    # every positive. Recall is undefined for queries with no positives, so
    # drop them — the bundle's `sample_passages` already enforces this
    # invariant for sampling, and we do the same here for loading.
    surviving_qids = set(bundle.qrels[bundle.qrels["relevance"] > 0]["qid"])
    if surviving_qids != set(bundle.queries["qid"]):
        cleaned_queries = bundle.queries[bundle.queries["qid"].isin(surviving_qids)].reset_index(
            drop=True
        )
        bundle = type(bundle)(
            name=bundle.name,
            passages=bundle.passages,
            queries=cleaned_queries,
            qrels=bundle.qrels[bundle.qrels["qid"].isin(surviving_qids)].reset_index(drop=True),
            metadata=bundle.metadata,
        )
    return bundle


def _stream_sample_corpus(
    stream: Iterator[BeirRecord],
    judged_pids: set[str],
    sample_size: int,
    seed: int,
) -> Iterator[BeirRecord]:
    """Yield judged passages (always) + a deterministic subsample of unjudged.

    Memory bound: O(sample_size). Deterministic for a given (seed, judged
    set, set of pids appearing in the corpus) — independent of stream order,
    so reordered or shuffled streams produce the same sample.

    Each unjudged pid gets a deterministic priority `H(seed | pid)` and the
    bottom-K priorities are kept (K = sample_size - |judged_emitted|). The
    quota is sized off *emitted* judged pids rather than *referenced* ones
    so orphan qrels (judged pid not present in corpus) do not silently
    shrink the result. Duplicate unjudged rows in the source are
    deduplicated before consuming a slot.
    """
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    import heapq  # noqa: PLC0415  -- localized helper

    judged_emitted: dict[str, BeirRecord] = {}  # pid -> first encounter
    unjudged_heap: list[tuple[int, str, BeirRecord]] = []  # max-heap by -priority
    unjudged_in_heap: set[str] = set()

    for rec in stream:
        if rec.pid in judged_pids:
            # Always remember (dedupe on first encounter); emission happens
            # after the stream so we can size the unjudged quota correctly.
            judged_emitted.setdefault(rec.pid, rec)
            continue
        if rec.pid in unjudged_in_heap:
            continue  # already a candidate, skip the duplicate
        priority = _stable_priority(rec.pid, seed)
        if len(unjudged_heap) < sample_size:
            heapq.heappush(unjudged_heap, (-priority, rec.pid, rec))
            unjudged_in_heap.add(rec.pid)
        elif priority < -unjudged_heap[0][0]:
            _, evicted_pid, _ = heapq.heapreplace(unjudged_heap, (-priority, rec.pid, rec))
            unjudged_in_heap.discard(evicted_pid)
            unjudged_in_heap.add(rec.pid)

    if len(judged_emitted) >= sample_size:
        # More judged than the quota — keep the top `sample_size` by the
        # same hashed-priority scheme so the choice is deterministic.
        ranked = sorted(
            judged_emitted.values(),
            key=lambda r: (_stable_priority(r.pid, seed), r.pid),
        )
        yield from ranked[:sample_size]
        return

    yield from judged_emitted.values()
    remaining = sample_size - len(judged_emitted)
    unjudged_sorted = sorted(unjudged_heap, key=lambda x: (-x[0], x[1]))
    for _, _, rec in unjudged_sorted[:remaining]:
        yield rec


def _stable_priority(pid: str, seed: int) -> int:
    """Deterministic 64-bit priority for a passage id under a sampling seed."""
    digest = hashlib.blake2b(f"{seed}|{pid}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def load_msmarco(
    sample_size: int = 100_000,
    seed: int = 42,
    cache_dir: str | Path | None = None,
    split: str = "dev",
) -> CorpusBundle:
    """Convenience wrapper over `load_beir_dataset("msmarco")`.

    MS-MARCO test qrels are not public; the BeIR `dev` split is the standard
    eval set and is the right default here.
    """
    return load_beir_dataset(
        dataset_name="msmarco",
        split=split,
        sample_size=sample_size,
        seed=seed,
        cache_dir=cache_dir,
    )


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def _first_present(raw: dict[str, Any], *keys: str) -> Any:
    """Return the first value among `keys` that is not missing/None.

    Note: `0`, `0.0`, and `""` are valid values — only `None` and absent
    keys are treated as missing. (Many BEIR/MS-MARCO ids start at 0.)
    """
    for k in keys:
        if k in raw and raw[k] is not None:
            return raw[k]
    return None


def _coerce_record(raw: BeirRecord | dict[str, Any]) -> BeirRecord:
    if isinstance(raw, BeirRecord):
        return raw
    pid_raw = _first_present(raw, "_id", "pid", "id")
    if pid_raw is None or str(pid_raw) == "":
        raise ValueError(f"BEIR corpus record missing id: {raw!r}")
    pid = str(pid_raw)
    title = str(raw.get("title") or "").strip()
    body = str(raw.get("text") or raw.get("body") or "").strip()
    text = f"{title}. {body}".strip(". ").strip() if title else body
    return BeirRecord(pid=pid, text=text)


def _coerce_query(raw: BeirQuery | dict[str, Any]) -> BeirQuery:
    if isinstance(raw, BeirQuery):
        return raw
    qid_raw = _first_present(raw, "_id", "qid", "id")
    if qid_raw is None or str(qid_raw) == "":
        raise ValueError(f"BEIR query missing id: {raw!r}")
    text = str(raw.get("text") or raw.get("query") or "").strip()
    return BeirQuery(qid=str(qid_raw), text=text)


def _coerce_qrel(raw: BeirQrel | dict[str, Any]) -> BeirQrel:
    if isinstance(raw, BeirQrel):
        return raw
    qid_raw = _first_present(raw, "query-id", "qid", "query_id")
    pid_raw = _first_present(raw, "corpus-id", "pid", "doc_id")
    if qid_raw is None or pid_raw is None or str(qid_raw) == "" or str(pid_raw) == "":
        raise ValueError(f"BEIR qrel missing qid or pid: {raw!r}")
    score_raw = _first_present(raw, "score", "relevance")
    relevance = float(score_raw) if score_raw is not None else 0.0
    if relevance < 0:
        raise ValueError(f"BEIR qrel has negative relevance: {raw!r}")
    return BeirQrel(qid=str(qid_raw), pid=str(pid_raw), relevance=relevance)


__all__ = [
    "BeirQrel",
    "BeirQuery",
    "BeirRecord",
    "bundle_from_beir_records",
    "load_beir_dataset",
    "load_msmarco",
]
