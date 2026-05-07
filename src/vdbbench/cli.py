"""CLI entry point — wires the corpus / encoder / bench / plot pipelines."""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console

from vdbbench import __version__

if TYPE_CHECKING:  # pragma: no cover - import only resolved by type-checker
    from vdbbench.bench import BenchSpec
    from vdbbench.embed import EncodedBundle

app = typer.Typer(
    name="vdbbench",
    help="Reproducible side-by-side benchmark of pgvector, Qdrant, LanceDB, and Chroma.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


# Exit code 2 is the standard "user-facing error" code (see bash + the
# typer/click convention); 1 is reserved for "the bench ran but every
# adapter failed", 0 is success.
_USER_ERROR_EXIT = 2

# Exception types we know belong to the "user typo / missing file /
# wrong value" bucket — for these we print the message verbatim and
# exit 2 instead of dumping a traceback. Anything outside this tuple
# is a real bug and should still raise so the stack is preserved.
_USER_FACING_ERRORS: tuple[type[BaseException], ...] = (
    FileNotFoundError,
    ValueError,
)


def _handle_user_error(exc: BaseException) -> None:
    """Print a clean error line and exit with the standard user-error code.

    Called from each command's outer try/except. ``typer.Exit`` is what
    the rest of the CLI uses to signal exit codes through Typer's
    runtime, so reuse it for consistency.
    """
    name = type(exc).__name__
    console.print(f"[red]error[/] ({name}): {exc}")
    raise typer.Exit(code=_USER_ERROR_EXIT) from exc


# `--adapter` choices: the four production backends + an in-process
# brute-force "exact" adapter that's also exposed as "memory" so users
# can ask for it by either name. Anything outside this set is a typo,
# not a quiet skip.
_ADAPTER_CHOICES: frozenset[str] = frozenset(
    {"pgvector", "qdrant", "lancedb", "chroma", "exact", "memory"}
)


def _resolve_adapter_names(names: list[str] | None) -> set[str]:
    """Validate ``--adapter`` values and fold the ``memory`` alias.

    Returns a normalised set with ``memory`` rewritten to ``exact`` so
    the spec-building branches only have to check one name.
    """
    if not names:
        return set()
    resolved: set[str] = set()
    for raw in names:
        name = raw.strip().lower()
        if name not in _ADAPTER_CHOICES:
            raise typer.BadParameter(
                f"unknown adapter {raw!r}; expected one of "
                f"{sorted(_ADAPTER_CHOICES)}",
                param_hint="--adapter",
            )
        if name == "memory":
            name = "exact"
        resolved.add(name)
    return resolved


def _reject_unselected_adapter_flags(
    selected: set[str],
    *,
    pgvector_dsn: str | None,
    qdrant_url: str | None,
    lancedb_path: Path | None,
    chroma_path: Path | None,
) -> None:
    """Refuse per-adapter flags for adapters the user didn't pick via --adapter.

    Without this, ``--adapter pgvector --qdrant-url http://...`` silently
    runs both because each URL/path flag used to build its own spec
    regardless of ``--adapter``. The selection is authoritative — a flag
    pointing at an unselected adapter is a contradiction.
    """
    contradictions: list[tuple[str, str]] = []
    if pgvector_dsn is not None and "pgvector" not in selected:
        contradictions.append(("pgvector", "--pgvector-dsn"))
    if qdrant_url is not None and "qdrant" not in selected:
        contradictions.append(("qdrant", "--qdrant-url"))
    if lancedb_path is not None and "lancedb" not in selected:
        contradictions.append(("lancedb", "--lancedb-path"))
    if chroma_path is not None and "chroma" not in selected:
        contradictions.append(("chroma", "--chroma-path"))
    if not contradictions:
        return
    # One ValueError listing all conflicts, so the user fixes them in
    # one round-trip rather than discovering them one at a time.
    parts = [
        f"{flag} passed but {adapter!r} not in --adapter selection"
        for adapter, flag in contradictions
    ]
    selection_list = ", ".join(sorted(selected)) or "(none)"
    raise ValueError(
        "; ".join(parts)
        + f". --adapter selection: {selection_list}. Either add the missing "
        "adapter(s) to --adapter or drop the conflicting flag(s)."
    )


def _apply_adapter_selection(
    selected: set[str],
    *,
    pgvector_dsn: str | None,
    qdrant_url: str | None,
    lancedb_path: Path | None,
    chroma_path: Path | None,
) -> tuple[str | None, str | None, Path | None, Path | None]:
    """Reduce the per-adapter knobs to only those the user picked via --adapter.

    When the user passes ``--adapter``, that selection is authoritative —
    any per-adapter URL/path flag for an *unselected* adapter is a
    contradiction, not a quiet "build a spec for that adapter too".
    Adapters in the selection that are missing creds get the standard
    localhost defaults; adapters outside the selection have their creds
    cleared so :func:`_build_bench_specs` can stay a pure data wiring
    helper.
    """
    _reject_unselected_adapter_flags(
        selected,
        pgvector_dsn=pgvector_dsn,
        qdrant_url=qdrant_url,
        lancedb_path=lancedb_path,
        chroma_path=chroma_path,
    )
    if "pgvector" in selected and pgvector_dsn is None:
        pgvector_dsn = "postgresql://bench:bench@localhost:5433/bench"
    if "qdrant" in selected and qdrant_url is None:
        qdrant_url = "http://localhost:6333"
    if "lancedb" in selected and lancedb_path is None:
        lancedb_path = Path("data/lancedb")
    if "chroma" in selected and chroma_path is None:
        chroma_path = Path("data/chroma")
    if "pgvector" not in selected:
        pgvector_dsn = None
    if "qdrant" not in selected:
        qdrant_url = None
    if "lancedb" not in selected:
        lancedb_path = None
    if "chroma" not in selected:
        chroma_path = None
    return pgvector_dsn, qdrant_url, lancedb_path, chroma_path


def _apply_all_defaults(
    pgvector_dsn: str | None,
    qdrant_url: str | None,
    lancedb_path: Path | None,
    chroma_path: Path | None,
) -> tuple[str | None, str | None, Path | None, Path | None]:
    """Populate the per-adapter knobs that `--all` should default in.

    Service adapters (pgvector / qdrant) get the standard local URLs.
    Embedded adapters (lancedb / chroma) only get a path when the wheel
    is actually importable on the host — otherwise the user would see
    a confusing ImportError instead of a "skipped" message. Explicit
    caller values still win.
    """
    if pgvector_dsn is None:
        pgvector_dsn = "postgresql://bench:bench@localhost:5433/bench"
    if qdrant_url is None:
        qdrant_url = "http://localhost:6333"
    if lancedb_path is None:
        try:
            import lancedb  # noqa: F401, PLC0415

            lancedb_path = Path("data/lancedb")
        except ImportError:
            console.print("[yellow]--all:[/] skipping lancedb (no wheel for this platform)")
    if chroma_path is None:
        try:
            import chromadb  # noqa: F401, PLC0415

            chroma_path = Path("data/chroma")
        except ImportError:
            console.print("[yellow]--all:[/] skipping chroma (no wheel for this platform)")
    return pgvector_dsn, qdrant_url, lancedb_path, chroma_path


@app.command()
def version() -> None:
    """Print the installed vdbbench version."""
    console.print(f"vdbbench {__version__}")


@app.command()
def prep(
    out: Path = typer.Option(Path("data/encoded"), help="Where to write the encoded bundle."),
    dataset: str = typer.Option(
        "synthetic",
        help=(
            "Corpus source: 'synthetic', 'msmarco', 'wikipedia', "
            "or any BeIR/<name> identifier."
        ),
    ),
    sample_size: int = typer.Option(
        5_000,
        help=(
            "Number of passages to keep (synthetic = exact, BEIR = "
            "sub-sample). Default 5000 matches the demo bundle "
            "committed under results/demo/, so a bare `vdbbench prep` "
            "produces the same scale the README quotes; override via "
            "`--sample-size` for larger sweeps. "
            "Required for --dataset wikipedia."
        ),
    ),
    encoder: str | None = typer.Option(
        None,
        "--encoder",
        help=(
            "Encoder shortname: 'bge' (384-dim, default) or 'nomic' "
            "(768-dim, uses trust_remote_code=True). "
            "Mutually exclusive with --embed-model."
        ),
    ),
    embed_model: str | None = typer.Option(
        None,
        help=(
            "sentence-transformers model id (raw HuggingFace ID). "
            "Mutually exclusive with --encoder. "
            "Defaults to 'BAAI/bge-small-en-v1.5' when neither flag is set."
        ),
    ),
    dim: int = typer.Option(384, help="Embedding dim (only used for 'synthetic')."),
    seed: int = typer.Option(42, help="Random seed for deterministic sampling."),
    n_queries: int = typer.Option(100, help="Number of queries (only used for 'synthetic')."),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help=(
            "Maximum number of documents to stream (wikipedia only). "
            "Required when --dataset wikipedia; omitting it errors out "
            "with a helpful message pointing at --limit."
        ),
    ),
    fixtures: Path | None = typer.Option(
        None,
        "--fixtures",
        help=(
            "Path to a JSONL fixture file for --dataset wikipedia. "
            "When set, reads from the file instead of streaming from "
            "HuggingFace (useful for tests and offline smoke runs)."
        ),
    ),
    wiki_chunk_size: int = typer.Option(
        512,
        "--wiki-chunk-size",
        help="Wikipedia chunker: max chunk length in characters.",
    ),
    wiki_overlap: int = typer.Option(
        64,
        "--wiki-overlap",
        help="Wikipedia chunker: overlap in characters between adjacent chunks.",
    ),
) -> None:
    """Build corpus + embeddings + ground-truth qrels into a parquet bundle."""
    if encoder is not None and embed_model is not None:
        raise typer.BadParameter(
            "--encoder and --embed-model are mutually exclusive; pass one or the other.",
            param_hint="--encoder / --embed-model",
        )
    try:
        _prep_impl(
            out=out,
            dataset=dataset,
            sample_size=sample_size,
            encoder=encoder,
            embed_model=embed_model,
            dim=dim,
            seed=seed,
            n_queries=n_queries,
            limit=limit,
            fixtures=fixtures,
            wiki_chunk_size=wiki_chunk_size,
            wiki_overlap=wiki_overlap,
        )
    except _USER_FACING_ERRORS as exc:
        _handle_user_error(exc)


def _prep_impl(
    *,
    out: Path,
    dataset: str,
    sample_size: int,
    encoder: str | None,
    embed_model: str | None,
    dim: int,
    seed: int,
    n_queries: int,
    limit: int | None = None,
    fixtures: Path | None = None,
    wiki_chunk_size: int = 512,
    wiki_overlap: int = 64,
) -> None:
    from vdbbench.corpus import (  # noqa: PLC0415
        SyntheticConfig,
        generate_synthetic,
        load_beir_dataset,
        load_msmarco,
    )
    from vdbbench.embed import EncodedBundle, encode_corpus  # noqa: PLC0415

    if dataset == "synthetic":
        cfg = SyntheticConfig(n_passages=sample_size, n_queries=n_queries, dim=dim, seed=seed)
        synth = generate_synthetic(cfg)
        # Use the synthetic generator's *own* vectors — its qrels are the
        # brute-force top-k of these exact vectors, so re-encoding the text
        # would produce a recall-0 mismatch.
        encoded = EncodedBundle(
            bundle=synth.bundle,
            passage_vectors=synth.passage_vectors,
            query_vectors=synth.query_vectors,
            encoder_name="synthetic-gaussian",
            metadata={"dataset": "synthetic", "seed": seed, "dim": dim},
        )
        console.print(f"[green]synthetic[/] corpus: {encoded.bundle.n_passages} passages")
    elif dataset == "wikipedia":
        encoded = _prep_wikipedia(
            out=out,
            limit=limit,
            fixtures=fixtures,
            encoder=encoder,
            embed_model=embed_model,
            wiki_chunk_size=wiki_chunk_size,
            wiki_overlap=wiki_overlap,
        )
    else:
        if dataset == "msmarco":
            bundle = load_msmarco(sample_size=sample_size, seed=seed)
        else:
            bundle = load_beir_dataset(dataset_name=dataset, sample_size=sample_size, seed=seed)
        if encoder is not None:
            # Short name path: resolve via registry so trust_remote_code is
            # automatically propagated (critical for nomic).
            from vdbbench.embed.registry import build_encoder  # noqa: PLC0415

            enc = build_encoder(encoder)
        else:
            # Raw HF ID path (legacy --embed-model or the default).
            resolved_model = embed_model or "BAAI/bge-small-en-v1.5"
            if embed_model is None:
                # Neither flag was passed — keep existing default behaviour.
                pass
            else:
                # --embed-model with a raw HF ID: trust_remote_code stays
                # False. Users who need it for nomic should use --encoder nomic.
                console.print(
                    "[yellow]note[/]: --embed-model uses trust_remote_code=False. "
                    "For nomic-embed-text-v1.5 use --encoder nomic instead."
                )
            from vdbbench.embed import SentenceTransformerEncoder  # noqa: PLC0415

            enc = SentenceTransformerEncoder(model_name=resolved_model)
        encoded = encode_corpus(bundle, enc, metadata={"dataset": dataset, "seed": seed})
    encoded.save(out)
    console.print(f"[green]wrote[/] encoded bundle to {out}")


def _prep_wikipedia(
    *,
    out: Path,
    limit: int | None,
    fixtures: Path | None,
    encoder: str | None,
    embed_model: str | None,
    wiki_chunk_size: int,
    wiki_overlap: int,
) -> EncodedBundle:
    """Build an encoded bundle from Wikipedia articles.

    When ``fixtures`` is set, reads from the JSONL file instead of streaming
    from HuggingFace (offline/test path). Otherwise streams from HuggingFace
    using the ``datasets`` library (requires ``[wiki]`` extras).

    Wikipedia has no query/qrel ground truth, so we synthesise a minimal
    placeholder so the rest of the pipeline (which expects a full
    ``CorpusBundle``) works.  Concretely, we produce one synthetic query per
    1000 chunks (or at least one query) whose text is taken from the chunk
    title, with a self-relevance qrel of 1.0. This is adequate for
    ``vdbbench bench`` smoke runs but not for precision/recall evaluation
    against Wikipedia-specific qrels.
    """
    import pandas as pd  # noqa: PLC0415

    from vdbbench.corpus.bundle import CorpusBundle  # noqa: PLC0415
    from vdbbench.corpus.wikipedia import WikipediaCorpus, iter_chunks  # noqa: PLC0415
    from vdbbench.embed import encode_corpus  # noqa: PLC0415

    if fixtures is not None:
        if not fixtures.is_file():
            raise FileNotFoundError(f"Wikipedia fixtures file not found: {fixtures}")
        docs_iter = WikipediaCorpus.from_fixtures(fixtures)
        effective_limit = limit  # still honour --limit for truncation
        if effective_limit is not None:
            import itertools  # noqa: PLC0415

            docs_iter = itertools.islice(docs_iter, effective_limit)
        console.print(f"[green]wikipedia[/] loading fixtures from {fixtures}")
    else:
        if limit is None:
            raise ValueError(
                "Wikipedia is large; specify --limit (e.g. --limit 10000) for a sized run. "
                "To use a fixture file instead of streaming, pass --fixtures <path>."
            )
        console.print(f"[green]wikipedia[/] streaming {limit} articles from HuggingFace")
        docs_iter = WikipediaCorpus.from_huggingface(limit=limit)

    chunks = list(iter_chunks(docs_iter, chunk_size=wiki_chunk_size, overlap=wiki_overlap))
    if not chunks:
        raise ValueError("Wikipedia corpus produced zero chunks — check the fixture file or limit.")

    console.print(f"[green]wikipedia[/] {len(chunks)} chunks from articles")

    # Build a minimal CorpusBundle: passages = chunks, queries = one per
    # 1000 chunks (or at least 1), self-relevant qrels.
    passage_rows = [{"pid": c.chunk_id, "text": (c.title + " " + c.text).strip()} for c in chunks]
    passages = pd.DataFrame(passage_rows)

    query_stride = max(1, len(chunks) // max(1, len(chunks) // 1000))
    query_indices = list(range(0, len(chunks), query_stride))
    query_rows = [
        {"qid": f"wq_{i}", "text": chunks[i].title or chunks[i].text[:120]}
        for i in query_indices
    ]
    queries = pd.DataFrame(query_rows)

    qrel_rows = [
        {"qid": f"wq_{i}", "pid": chunks[i].chunk_id, "relevance": 1.0}
        for i in query_indices
    ]
    qrels = pd.DataFrame(qrel_rows)

    bundle = CorpusBundle(
        name=f"wikipedia@n={len(chunks)}",
        passages=passages,
        queries=queries,
        qrels=qrels,
        metadata={
            "dataset": "wikipedia",
            "n_chunks": len(chunks),
            "chunk_size": wiki_chunk_size,
            "overlap": wiki_overlap,
            "fixtures": str(fixtures) if fixtures else None,
        },
    )

    if encoder is not None:
        from vdbbench.embed.registry import build_encoder  # noqa: PLC0415

        enc = build_encoder(encoder)
    else:
        resolved_model = embed_model or "BAAI/bge-small-en-v1.5"
        if embed_model is not None:
            console.print(
                "[yellow]note[/]: --embed-model uses trust_remote_code=False. "
                "For nomic-embed-text-v1.5 use --encoder nomic instead."
            )
        from vdbbench.embed import SentenceTransformerEncoder  # noqa: PLC0415

        enc = SentenceTransformerEncoder(model_name=resolved_model)

    return encode_corpus(bundle, enc, metadata={"dataset": "wikipedia"})


def _build_bench_specs(
    *,
    pgvector_dsn: str | None,
    qdrant_url: str | None,
    lancedb_path: Path | None,
    chroma_path: Path | None,
    run_exact: bool,
    k: int,
    repeats: int,
    profile: str,
) -> list[BenchSpec]:
    """Translate the per-adapter CLI flags into a list of BenchSpec rows.

    Extracted from ``bench()`` so the command body stays under the
    branch-count lint cap; pure data wiring with no I/O.
    """
    from vdbbench.bench import BenchSpec  # noqa: PLC0415

    specs: list[BenchSpec] = []
    if pgvector_dsn:
        from vdbbench.adapters import PgVectorAdapter  # noqa: PLC0415

        specs.append(
            BenchSpec(
                adapter=PgVectorAdapter(dsn=pgvector_dsn),
                params={"index": "hnsw", "metric": "cosine"},
                k=k,
                repeats=repeats,
                label="pgvector:hnsw-default",
                profile=profile,
            )
        )
    if qdrant_url:
        from vdbbench.adapters import QdrantAdapter  # noqa: PLC0415

        specs.append(
            BenchSpec(
                adapter=QdrantAdapter(url=qdrant_url),
                params={"metric": "cosine"},
                k=k,
                repeats=repeats,
                label="qdrant:hnsw-default",
                profile=profile,
            )
        )
    if lancedb_path:
        from vdbbench.adapters import LanceDBAdapter  # noqa: PLC0415

        specs.append(
            BenchSpec(
                adapter=LanceDBAdapter(path=lancedb_path),
                params={"index": "ivf_pq", "metric": "cosine"},
                k=k,
                repeats=repeats,
                label="lancedb:ivf_pq-default",
                profile=profile,
            )
        )
    if chroma_path:
        from vdbbench.adapters import ChromaAdapter  # noqa: PLC0415

        specs.append(
            BenchSpec(
                adapter=ChromaAdapter(path=chroma_path),
                params={"metric": "cosine"},
                k=k,
                repeats=repeats,
                label="chroma:default",
                profile=profile,
            )
        )
    if run_exact:
        from vdbbench.adapters import ExactAdapter  # noqa: PLC0415

        specs.append(
            BenchSpec(
                adapter=ExactAdapter(),
                params={"metric": "cosine"},
                k=k,
                repeats=repeats,
                label="exact:bruteforce",
                profile=profile,
            )
        )
    return specs


@app.command()
def bench(
    encoded: Path = typer.Option(Path("data/encoded"), help="Encoded bundle directory."),
    out: Path | None = typer.Option(
        None,
        help=(
            "Where to write timings + summary parquet. Defaults to a "
            "timestamped subdir under results/run/ "
            "(``results/run/<UTC-isoformat>-<short-uuid>/``) so two "
            "parallel `vdbbench bench` invocations don't overwrite each "
            "other's output. The repo's top-level results/ tree still "
            "only holds named scale subdirs (results/demo/, "
            "results/100k/, results/full/, ...); pass --out explicitly "
            "to land outside results/run/ — `make bench-demo` / "
            "`make bench-100k` do exactly that for their conventional "
            "homes."
        ),
    ),
    pgvector_dsn: str | None = typer.Option(
        None, help="If set, run the pgvector adapter against this DSN."
    ),
    qdrant_url: str | None = typer.Option(
        None, help="If set, run the qdrant adapter against this URL."
    ),
    lancedb_path: Path | None = typer.Option(
        None, help="If set, run the lancedb adapter against this directory."
    ),
    chroma_path: Path | None = typer.Option(
        None, help="If set, run the chroma adapter against this directory."
    ),
    all_adapters: bool = typer.Option(
        False,
        "--all",
        help=(
            "Run every adapter the local install supports with sensible defaults: "
            "pgvector at the standard local DSN, qdrant at the standard local URL, "
            "and lancedb / chroma at data/lancedb / data/chroma if their wheels are "
            "importable. Equivalent to passing every --*-dsn / --*-path flag."
        ),
    ),
    adapter: list[str] | None = typer.Option(
        None,
        "--adapter",
        help=(
            "Pick adapters by name (repeatable). Choices: pgvector, qdrant, "
            "lancedb, chroma, exact, memory ('memory' is an alias for "
            "'exact', the in-process brute-force adapter — handy for "
            "smoke runs without Docker). Service adapters use the same "
            "defaults as --all (localhost DSN / URL); embedded adapters "
            "use data/lancedb / data/chroma. Combine with --pgvector-dsn / "
            "--qdrant-url / --lancedb-path / --chroma-path to override "
            "individual endpoints."
        ),
    ),
    k: int = typer.Option(10, help="Top-k for retrieval."),
    repeats: int = typer.Option(
        -1,
        help=(
            "Number of measured query passes per spec. The default sentinel -1 "
            "means 'use the profile default' (cold/warm => 1, p99 => 5); pass an "
            "explicit positive value to override."
        ),
    ),
    profile: str = typer.Option(
        "warm",
        help="Bench profile: 'cold' (no warm-up), 'warm' (default), 'p99' (50 warm-up + 5 repeats).",
    ),
    prometheus_port: int | None = typer.Option(
        None,
        "--prometheus-port",
        help=(
            "If set, start a Prometheus HTTP scrape endpoint on this port "
            "before the bench begins. After the run: "
            "``curl localhost:<PORT>/metrics | grep vdbbench`` shows "
            "vdbbench_ingest_vectors_total, vdbbench_query_latency_seconds, "
            "vdbbench_query_recall_at_k, and vdbbench_bench_duration_seconds. "
            "Unset (default) means no exporter is started."
        ),
    ),
    profiler: str | None = typer.Option(
        None,
        "--profiler",
        help=(
            "Opt-in profiler to run alongside the bench.  Supported: "
            "'py-spy' (flame graph SVG, requires py-spy installed: "
            "``pip install 'vdbbench[profile]'``), 'iostat' (I/O stats "
            "text file using the system iostat binary), or 'ps_mem' "
            "(per-process memory breakdown — shared vs private vs swap — "
            "using the system ps_mem binary: brew install ps_mem or "
            "apt install ps_mem).  All three soft-fail with a warning "
            "when the binary is not on PATH.  Example: "
            "--profiler py-spy  or  --profiler iostat  or  --profiler ps_mem"
        ),
    ),
) -> None:
    """Run the bench across every adapter the user enabled by passing a DSN/path."""
    try:
        _bench_impl(
            encoded=encoded,
            out=out,
            pgvector_dsn=pgvector_dsn,
            qdrant_url=qdrant_url,
            lancedb_path=lancedb_path,
            chroma_path=chroma_path,
            all_adapters=all_adapters,
            adapter=adapter,
            k=k,
            repeats=repeats,
            profile=profile,
            prometheus_port=prometheus_port,
            profiler=profiler,
        )
    except _USER_FACING_ERRORS as exc:
        _handle_user_error(exc)


def _resolve_bench_out(out: Path | None) -> Path:
    """Resolve the bench --out path, auto-suffixing the default.

    When the user leaves --out unset, default to
    ``results/run/<UTC-iso>-<short-uuid>/`` so two parallel `vdbbench
    bench` invocations don't overwrite each other's parquet. Explicit
    --out (e.g. `--out results/demo`) is used verbatim — the Makefile
    targets that need a fixed location keep working unchanged.

    UTC isoformat is filename-safe modulo the colon, which we replace
    with `-` so the path round-trips through Windows shells too. The
    short uuid suffix breaks ties when two runs happen in the same
    second (CI matrices, scripts in tight loops).
    """
    if out is not None:
        return out
    import uuid as _uuid  # noqa: PLC0415
    from datetime import UTC, datetime  # noqa: PLC0415

    stamp = datetime.now(UTC).isoformat(timespec="seconds").replace(":", "-")
    short = _uuid.uuid4().hex[:8]
    return Path("results/run") / f"{stamp}-{short}"


_SUPPORTED_PROFILERS: frozenset[str] = frozenset({"py-spy", "iostat", "ps_mem"})


def _bench_impl(
    *,
    encoded: Path,
    out: Path | None,
    pgvector_dsn: str | None,
    qdrant_url: str | None,
    lancedb_path: Path | None,
    chroma_path: Path | None,
    all_adapters: bool,
    adapter: list[str] | None,
    k: int,
    repeats: int,
    profile: str,
    prometheus_port: int | None = None,
    profiler: str | None = None,
) -> None:
    from vdbbench.bench import run_bench  # noqa: PLC0415
    from vdbbench.embed import load_encoded_bundle  # noqa: PLC0415

    # Validate profiler choice early so the user gets a clean error.
    if profiler is not None and profiler not in _SUPPORTED_PROFILERS:
        raise ValueError(
            f"unknown profiler {profiler!r}; supported: {sorted(_SUPPORTED_PROFILERS)}"
        )

    out = _resolve_bench_out(out)
    console.print(f"[green]output[/]: {out}")

    _prom_server = None
    if prometheus_port is not None:
        from vdbbench import prom_metrics  # noqa: PLC0415

        _prom_server = prom_metrics.start_exporter(prometheus_port)
        console.print(
            f"[green]prometheus[/]: scraping at http://localhost:{prometheus_port}/metrics"
        )

    try:
        selected_adapters = _resolve_adapter_names(adapter)
        run_exact = "exact" in selected_adapters
        if all_adapters:
            pgvector_dsn, qdrant_url, lancedb_path, chroma_path = _apply_all_defaults(
                pgvector_dsn, qdrant_url, lancedb_path, chroma_path
            )
        if selected_adapters:
            pgvector_dsn, qdrant_url, lancedb_path, chroma_path = _apply_adapter_selection(
                selected_adapters,
                pgvector_dsn=pgvector_dsn,
                qdrant_url=qdrant_url,
                lancedb_path=lancedb_path,
                chroma_path=chroma_path,
            )

        enc = load_encoded_bundle(encoded)
        specs = _build_bench_specs(
            pgvector_dsn=pgvector_dsn,
            qdrant_url=qdrant_url,
            lancedb_path=lancedb_path,
            chroma_path=chroma_path,
            run_exact=run_exact,
            k=k,
            repeats=repeats,
            profile=profile,
        )

        if not specs:
            console.print(
                "[yellow]no adapters enabled[/] — pass at least one of "
                "--pgvector-dsn / --qdrant-url / --lancedb-path / --chroma-path, "
                "--adapter <name> (pgvector|qdrant|lancedb|chroma|exact|memory), "
                "or --all to use sensible defaults"
            )
            raise typer.Exit(code=2)

        console.print(f"[green]running[/] {len(specs)} specs against {enc.bundle.name}")
        # `--all` is the "best-effort across whatever's running" entry
        # point, so a single dead service must not kill the whole run.
        # Single-adapter mode (the user explicitly named one DB) keeps the
        # hard-fail behavior — there's no other adapter to keep going for.
        #
        # ``out=out`` wires the runner's incremental-write + partial-manifest
        # path to the CLI: a Ctrl-C mid-run flushes the specs that already
        # completed with ``partial=True`` stamped in ``bench_manifest.json``
        # before the exception propagates. Without this, ``out_path`` inside
        # the runner stayed ``None`` and the documented "Ctrl-C flushes a
        # partial manifest" behaviour was dead code from the CLI entry.
        # ``BenchResult.save(out)`` below is still the canonical final write
        # — it overwrites the per-spec incremental files with the final
        # ``partial=False`` manifest, so there's no double-write conflict.
        with _build_profiler_ctx(profiler=profiler, out=out):
            result = run_bench(enc, specs, progress=True, tolerate_failures=all_adapters, out=out)
        out_path = result.save(out)
        console.print(f"[green]wrote[/] timings + summary to {out_path}")
        n_ran = len(specs) - len(result.skipped)
        if result.skipped:
            for skip in result.skipped:
                console.print(
                    f"[yellow]skipped[/] {skip.label} ({skip.error_type}): {skip.reason}"
                )
            console.print(
                f"[yellow]bench --all summary:[/] {n_ran}/{len(specs)} adapters ran, "
                f"{len(result.skipped)} skipped"
            )
        if all_adapters and n_ran == 0:
            console.print("[red]bench --all:[/] every adapter failed; exiting non-zero")
            raise typer.Exit(code=1)
    finally:
        if _prom_server is not None:
            from vdbbench import prom_metrics  # noqa: PLC0415

            prom_metrics.stop_exporter(_prom_server)


@contextlib.contextmanager
def _build_profiler_ctx(
    profiler: str | None, out: Path
) -> Generator[None, None, None]:
    """Yield inside the requested profiler context, or yield a no-op."""
    if profiler == "py-spy":
        from vdbbench.profile.pyspy import PySpyProfiler  # noqa: PLC0415

        prof = PySpyProfiler(output_dir=out)
        if prof.available:
            console.print(f"[green]py-spy[/]: recording flame graph → {prof.svg_path}")
        with prof:
            yield
    elif profiler == "iostat":
        from vdbbench.profile.iostat import IostatProfiler  # noqa: PLC0415

        prof_io = IostatProfiler(output_dir=out)
        if prof_io.available:
            console.print(f"[green]iostat[/]: recording I/O stats → {prof_io.txt_path}")
        with prof_io:
            yield
    elif profiler == "ps_mem":
        from vdbbench.profile.ps_mem import PsMemProfiler  # noqa: PLC0415

        prof_mem = PsMemProfiler(output_dir=out)
        if prof_mem.available:
            console.print(
                f"[green]ps_mem[/]: recording per-process memory breakdown → {prof_mem.log_path}"
            )
        with prof_mem:
            yield
    else:
        yield


def _resolve_plot_summary(summary: Path | None) -> Path:
    """Resolve --summary, auto-picking the most recent results/run/ subdir.

    The bench default writes to ``results/run/<timestamp>-<short-uuid>/``
    so plot can no longer point at a fixed path. When --summary is
    unset, we look under ``results/run/`` for subdirs containing a
    ``summary.parquet`` and pick the most recently modified one — which
    matches what a user means by "plot the run I just did". If nothing
    is there, raise ValueError so the CLI prints a clean error instead
    of a FileNotFoundError on a path the user never typed.
    """
    if summary is not None:
        if summary.is_dir():
            raise FileNotFoundError(
                f"{summary} is a directory, not a parquet file. "
                f"Did you mean '{summary}/summary.parquet'?"
            )
        if not summary.is_file():
            # Infer a helpful "next command" hint from the path so users
            # who run `make plots` before any bench get an actionable
            # message rather than a raw FileNotFoundError traceback.
            # Best-effort: match common scale subdir names (100k, demo,
            # full); fall back to a generic hint for arbitrary paths.
            scale = summary.parent.name  # e.g. "100k", "demo", "full"
            known_scales = {"100k": "make bench-100k", "demo": "make bench-demo", "full": "make bench-1m"}
            suggestion = known_scales.get(scale, "vdbbench bench")
            raise FileNotFoundError(
                f"{summary} does not exist. "
                f"Run '{suggestion}' first to generate results."
            )
        return summary
    root = Path("results/run")
    if not root.is_dir():
        raise ValueError(
            "no results/run/ directory found; pass --summary <path> "
            "explicitly (e.g. --summary results/demo/summary.parquet)"
        )
    candidates = [
        sub / "summary.parquet"
        for sub in root.iterdir()
        if sub.is_dir() and (sub / "summary.parquet").is_file()
    ]
    if not candidates:
        raise ValueError(
            f"no summary.parquet found under {root}/; pass --summary "
            "<path> explicitly (e.g. --summary results/demo/summary.parquet)"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


@app.command()
def plot(
    summary: Path | None = typer.Option(
        None,
        "--summary",
        help=(
            "Path to bench summary parquet. When unset, picks the most "
            "recent ``results/run/<stamp>-<uuid>/summary.parquet`` so a "
            "bare `vdbbench bench && vdbbench plot` keeps working — but "
            "you'll get a clean error (rather than a missing-file "
            "traceback) if no run is there."
        ),
    ),
    out: Path = typer.Option(Path("assets"), help="Where to write the chart files."),
    baseline_label: str | None = typer.Option(
        None,
        "--baseline-label",
        help=(
            "Speedup chart baseline row, matched against the summary's `label` "
            "column (e.g. 'chroma:default'). Required when the baseline DB "
            "(chroma if present, else alphabetically-first) has multiple "
            "configs in the summary; the chart anchors that exact row at 1.0 "
            "and computes ratios off its p95."
        ),
    ),
) -> None:
    """Regenerate every standard chart from a bench summary."""
    try:
        resolved = _resolve_plot_summary(summary)
        _plot_impl(summary=resolved, out=out, baseline_label=baseline_label)
    except _USER_FACING_ERRORS as exc:
        _handle_user_error(exc)


def _plot_impl(*, summary: Path, out: Path, baseline_label: str | None) -> None:
    from vdbbench.plot import plot_all  # noqa: PLC0415

    # Capture UserWarning emitted by `plot_speedup_vs_baseline` (e.g. the
    # "no baseline_db given; defaulting to alphabetically-first" hint).
    # The library still emits warnings so programmatic callers can hook
    # them; the CLI layer turns them into a Rich-coloured advisory line
    # so the user gets a tidy `[yellow]note[/]: ...` instead of a
    # Python-style `UserWarning: ...` blob in stderr.
    #
    # Re-emit only warnings whose origin is ``vdbbench.plot.charts``.
    # The previous implementation re-rendered every captured UserWarning
    # as a vdbbench advisory, which laundered third-party warnings
    # (matplotlib / pandas / etc.) through the CLI's note channel and
    # hid messages the user genuinely needed to see.
    with warnings.catch_warnings(record=True) as caught:
        # Inside the block: capture every warning so we can decide what
        # to surface ourselves. Restoring on exit means pytest's
        # outer ``-W error`` filter is unaffected.
        warnings.simplefilter("always")
        paths = plot_all(summary, out, baseline_label=baseline_label)
    for w in caught:
        if issubclass(w.category, UserWarning) and getattr(w, "filename", "").endswith(
            "vdbbench/plot/charts.py"
        ):
            console.print(f"[yellow]note[/]: {w.message}")
    for name, (png, svg) in paths.items():
        console.print(f"[green]{name}[/]: {png.name} + {svg.name}")


def _parse_grid_option(grid_items: list[str]) -> dict[str, list[object]]:
    """Parse repeated ``--grid KEY=VAL,VAL,VAL`` options into a parameter grid dict.

    Each item must be ``KEY=VAL[,VAL...]``. Values are cast to int or float
    when the conversion is unambiguous; otherwise kept as strings.

    Examples::

        ["ef_search=32,64,128", "m=8,16"]
        -> {"ef_search": [32, 64, 128], "m": [8, 16]}
    """
    grid: dict[str, list[object]] = {}
    for item in grid_items:
        if "=" not in item:
            raise ValueError(
                f"--grid value {item!r} must be in KEY=VAL[,VAL,...] format "
                "(example: ef_search=32,64,128)"
            )
        key, raw_vals = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--grid item has an empty key: {item!r}")
        parsed: list[object] = []
        for raw_v in raw_vals.split(","):
            stripped = raw_v.strip()
            if not stripped:
                continue
            # Try int first, then float, then leave as string.
            try:
                parsed.append(int(stripped))
            except ValueError:
                try:
                    parsed.append(float(stripped))
                except ValueError:
                    parsed.append(stripped)
        if not parsed:
            raise ValueError(f"--grid item {item!r} has no values after '='")
        grid[key] = parsed
    return grid


@app.command()
def sweep(
    adapter: str = typer.Option(
        ...,
        "--adapter",
        help=(
            "Adapter to sweep. Currently only 'exact' / 'memory' run "
            "offline (no Docker). Choices: exact, memory."
        ),
    ),
    grid: list[str] = typer.Option(
        ...,
        "--grid",
        help=(
            "Grid axis in KEY=VAL,VAL,VAL format (repeatable). "
            "Example: --grid ef_search=32,64,128 --grid m=8,16 "
            "runs 6 trials (2 x 3). Every Cartesian product of all "
            "--grid axes is evaluated."
        ),
    ),
    encoded: Path = typer.Option(
        Path("data/encoded"),
        help="Encoded bundle directory (output of `vdbbench prep`).",
    ),
    out: Path = typer.Option(
        Path("results/sweep"),
        help="Directory to write sweep.parquet and sweep_pareto.{png,svg}.",
    ),
    top_k: int = typer.Option(10, help="Top-k for recall computation."),
    all_adapters: bool = typer.Option(
        False,
        "--all",
        help=(
            "Sweep all configured adapters (currently not implemented; "
            "mutually exclusive with --adapter)."
        ),
    ),
) -> None:
    """Sweep adapter index/search knobs across a Cartesian grid and plot the Pareto frontier."""
    if all_adapters:
        console.print("[red]error[/]: --all is not yet implemented for sweep; pass --adapter instead")
        raise typer.Exit(code=2)
    try:
        _sweep_impl(
            adapter=adapter,
            grid=grid,
            encoded=encoded,
            out=out,
            top_k=top_k,
        )
    except _USER_FACING_ERRORS as exc:
        _handle_user_error(exc)


def _sweep_impl(
    *,
    adapter: str,
    grid: list[str],
    encoded: Path,
    out: Path,
    top_k: int,
) -> None:
    import asyncio  # noqa: PLC0415

    from vdbbench.plot.charts import plot_sweep_pareto  # noqa: PLC0415
    from vdbbench.sweep import SweepSpec, run_sweep, write_sweep_parquet  # noqa: PLC0415

    parameter_grid = _parse_grid_option(grid)
    n_trials = 1
    for vals in parameter_grid.values():
        n_trials *= len(vals)

    console.print(
        f"[green]sweep[/]: adapter={adapter!r} grid={parameter_grid} "
        f"({n_trials} trial{'s' if n_trials != 1 else ''})"
    )

    spec = SweepSpec(
        adapter=adapter,
        parameter_grid=parameter_grid,
        dataset=str(encoded),
        top_k=top_k,
    )

    results = asyncio.run(run_sweep(spec))
    if not results:
        console.print("[yellow]warning[/]: sweep produced zero results")
        raise typer.Exit(code=1)

    out.mkdir(parents=True, exist_ok=True)
    parquet_path = write_sweep_parquet(results, out / "sweep.parquet")
    console.print(f"[green]wrote[/] {len(results)} trial(s) to {parquet_path}")

    png, svg = plot_sweep_pareto(parquet_path, out)
    console.print(f"[green]chart[/]: {png.name} + {svg.name}")


@app.command(name="plot-sweep")
def plot_sweep(
    sweep_parquet: Path = typer.Option(
        ...,
        "--sweep-parquet",
        help="Path to sweep.parquet written by `vdbbench sweep`.",
    ),
    out: Path = typer.Option(
        Path("results/sweep"),
        help="Directory to write sweep_pareto.{png,svg}.",
    ),
) -> None:
    """Read a sweep parquet and regenerate the Pareto-frontier scatter chart."""
    try:
        if not sweep_parquet.is_file():
            raise FileNotFoundError(
                f"{sweep_parquet} does not exist. Run 'vdbbench sweep' first."
            )
        from vdbbench.plot.charts import plot_sweep_pareto  # noqa: PLC0415

        png, svg = plot_sweep_pareto(sweep_parquet, out)
        console.print(f"[green]chart[/]: {png.name} + {svg.name}")
    except _USER_FACING_ERRORS as exc:
        _handle_user_error(exc)


@app.command(name="compare-encoders")
def compare_encoders(
    adapter: str = typer.Option(
        "exact",
        "--adapter",
        help=(
            "Adapter to bench against.  Use 'exact' / 'memory' for offline "
            "runs (no Docker).  Choices: exact, memory."
        ),
    ),
    encoders: str = typer.Option(
        "bge",
        "--encoders",
        help=(
            "Comma-separated encoder shortnames to compare.  Both must be "
            "registered in ENCODER_REGISTRY.  Example: bge,nomic.  "
            "Default: bge (nomic requires a ~500 MB model download)."
        ),
    ),
    dataset: str = typer.Option(
        "synthetic",
        "--dataset",
        help=(
            "Dataset shortname passed to the comparison runner.  "
            "Only 'synthetic' is supported offline (no HuggingFace download). "
            "Default: synthetic."
        ),
    ),
    out: Path = typer.Option(
        Path("results/encoder-compare"),
        "--out",
        help=(
            "Output directory.  Receives ``encoder_comparison.parquet`` and "
            "``encoder_comparison.png``.  Created if it does not exist."
        ),
    ),
    corpus_size: int = typer.Option(
        500,
        "--corpus-size",
        help=(
            "Number of corpus passages for the comparison run.  "
            "Default 500 — small enough for a fast offline smoke run."
        ),
    ),
    top_k: int = typer.Option(10, "--top-k", help="Retrieval top-k for recall computation."),
) -> None:
    """Bench multiple encoders side-by-side and produce a comparison parquet + chart."""
    try:
        _compare_encoders_impl(
            adapter=adapter,
            encoders=encoders,
            dataset=dataset,
            out=out,
            corpus_size=corpus_size,
            top_k=top_k,
        )
    except _USER_FACING_ERRORS as exc:
        _handle_user_error(exc)


def _compare_encoders_impl(
    *,
    adapter: str,
    encoders: str,
    dataset: str,
    out: Path,
    corpus_size: int,
    top_k: int,
) -> None:
    from vdbbench.encode.compare import (  # noqa: PLC0415
        EncoderComparisonSpec,
        plot_encoder_comparison,
        run_encoder_comparison,
    )

    encoder_list = [e.strip() for e in encoders.split(",") if e.strip()]
    if not encoder_list:
        raise ValueError("--encoders must be a non-empty comma-separated list")

    spec = EncoderComparisonSpec(
        adapter=adapter,
        dataset=dataset,
        top_k=top_k,
        encoders=tuple(encoder_list),
        corpus_size=corpus_size,
    )

    console.print(
        f"[green]compare-encoders[/]: adapter={adapter!r} "
        f"encoders={encoder_list} dataset={dataset!r} corpus_size={corpus_size}"
    )

    results = run_encoder_comparison(spec, output_dir=out)

    parquet_path = out / "encoder_comparison.parquet"
    png_path = out / "encoder_comparison.png"
    plot_encoder_comparison(parquet_path, png_path)

    console.print(f"[green]wrote[/] comparison parquet → {parquet_path}")
    console.print(f"[green]wrote[/] comparison chart  → {png_path}")
    for r in results:
        console.print(
            f"  {r.encoder}: recall@{top_k}={r.recall_at_k:.3f} "
            f"p95={r.p95_ms:.2f}ms dim={r.embedding_dim}"
        )


report_app = typer.Typer(
    name="report",
    help="Generate reports from bench results.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(report_app)


@report_app.command(name="html")
def report_html(
    from_dir: Path = typer.Option(
        ...,
        "--from",
        help=(
            "Results directory containing summary.parquet and bench_manifest.json "
            "(e.g. results/demo/)."
        ),
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        help="Output path for the generated HTML report (e.g. results/demo/report.html).",
    ),
    charts_dir: Path | None = typer.Option(
        None,
        "--charts-dir",
        help=(
            "Optional directory of PNG chart files (e.g. assets/) to embed inline. "
            "When omitted the charts section is skipped."
        ),
    ),
) -> None:
    """Render a self-contained HTML report from a bench results directory."""
    try:
        parquet_path = from_dir / "summary.parquet"
        manifest_path = from_dir / "bench_manifest.json"
        if not parquet_path.is_file():
            raise FileNotFoundError(
                f"{parquet_path} does not exist. "
                "Run 'vdbbench bench' (or 'make bench-demo') to generate results first."
            )
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"{manifest_path} does not exist. "
                "Run 'vdbbench bench' (or 'make bench-demo') to generate results first."
            )
        from vdbbench.report.html import render_html_report  # noqa: PLC0415

        render_html_report(
            parquet_path=parquet_path,
            manifest_path=manifest_path,
            output_path=out,
            charts_dir=charts_dir,
        )
        console.print(f"[green]wrote[/] HTML report to {out}")
    except _USER_FACING_ERRORS as exc:
        _handle_user_error(exc)


def main() -> None:  # pragma: no cover - thin wrapper
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
