"""CLI entry point — wires the corpus / encoder / bench / plot pipelines."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console

from vdbbench import __version__

if TYPE_CHECKING:  # pragma: no cover - import only resolved by type-checker
    from vdbbench.bench import BenchSpec

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
        help="Corpus source: 'synthetic', 'msmarco', or any BeIR/<name> identifier.",
    ),
    sample_size: int = typer.Option(
        5_000,
        help=(
            "Number of passages to keep (synthetic = exact, BEIR = "
            "sub-sample). Default 5000 matches the demo bundle "
            "committed under results/demo/, so a bare `vdbbench prep` "
            "produces the same scale the README quotes; override via "
            "`--sample-size` for larger sweeps."
        ),
    ),
    embed_model: str = typer.Option(
        "BAAI/bge-small-en-v1.5", help="sentence-transformers model id."
    ),
    dim: int = typer.Option(384, help="Embedding dim (only used for 'synthetic')."),
    seed: int = typer.Option(42, help="Random seed for deterministic sampling."),
    n_queries: int = typer.Option(100, help="Number of queries (only used for 'synthetic')."),
) -> None:
    """Build corpus + embeddings + ground-truth qrels into a parquet bundle."""
    try:
        _prep_impl(
            out=out,
            dataset=dataset,
            sample_size=sample_size,
            embed_model=embed_model,
            dim=dim,
            seed=seed,
            n_queries=n_queries,
        )
    except _USER_FACING_ERRORS as exc:
        _handle_user_error(exc)


def _prep_impl(
    *,
    out: Path,
    dataset: str,
    sample_size: int,
    embed_model: str,
    dim: int,
    seed: int,
    n_queries: int,
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
    else:
        if dataset == "msmarco":
            bundle = load_msmarco(sample_size=sample_size, seed=seed)
        else:
            bundle = load_beir_dataset(dataset_name=dataset, sample_size=sample_size, seed=seed)
        from vdbbench.embed import SentenceTransformerEncoder  # noqa: PLC0415

        encoder = SentenceTransformerEncoder(model_name=embed_model)
        encoded = encode_corpus(bundle, encoder, metadata={"dataset": dataset, "seed": seed})
    encoded.save(out)
    console.print(f"[green]wrote[/] encoded bundle to {out}")


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
) -> None:
    from vdbbench.bench import run_bench  # noqa: PLC0415
    from vdbbench.embed import load_encoded_bundle  # noqa: PLC0415

    out = _resolve_bench_out(out)
    console.print(f"[green]output[/]: {out}")

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
    result = run_bench(enc, specs, progress=True, tolerate_failures=all_adapters)
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
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        paths = plot_all(summary, out, baseline_label=baseline_label)
    for w in caught:
        if issubclass(w.category, UserWarning):
            console.print(f"[yellow]note[/]: {w.message}")
    for name, (png, svg) in paths.items():
        console.print(f"[green]{name}[/]: {png.name} + {svg.name}")


def main() -> None:  # pragma: no cover - thin wrapper
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
