"""CLI entry point — wires the corpus / encoder / bench / plot pipelines."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from vdbbench import __version__

app = typer.Typer(
    name="vdbbench",
    help="Reproducible side-by-side benchmark of pgvector, Qdrant, LanceDB, and Chroma.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


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
        10_000, help="Number of passages to keep (synthetic = exact, BEIR = sub-sample)."
    ),
    embed_model: str = typer.Option(
        "BAAI/bge-small-en-v1.5", help="sentence-transformers model id."
    ),
    dim: int = typer.Option(384, help="Embedding dim (only used for 'synthetic')."),
    seed: int = typer.Option(42, help="Random seed for deterministic sampling."),
    n_queries: int = typer.Option(100, help="Number of queries (only used for 'synthetic')."),
) -> None:
    """Build corpus + embeddings + ground-truth qrels into a parquet bundle."""
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


@app.command()
def bench(
    encoded: Path = typer.Option(Path("data/encoded"), help="Encoded bundle directory."),
    out: Path = typer.Option(Path("results"), help="Where to write timings + summary parquet."),
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
    from vdbbench.bench import BenchSpec, run_bench  # noqa: PLC0415
    from vdbbench.embed import load_encoded_bundle  # noqa: PLC0415

    enc = load_encoded_bundle(encoded)
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

    if not specs:
        console.print(
            "[yellow]no adapters enabled[/] — pass at least one of "
            "--pgvector-dsn / --qdrant-url / --lancedb-path / --chroma-path"
        )
        raise typer.Exit(code=2)

    console.print(f"[green]running[/] {len(specs)} specs against {enc.bundle.name}")
    result = run_bench(enc, specs, progress=True)
    out_path = result.save(out)
    console.print(f"[green]wrote[/] timings + summary to {out_path}")


@app.command()
def plot(
    summary: Path = typer.Option(
        Path("results/summary.parquet"), help="Path to bench summary parquet."
    ),
    out: Path = typer.Option(Path("assets"), help="Where to write the chart files."),
) -> None:
    """Regenerate every standard chart from a bench summary."""
    from vdbbench.plot import plot_all  # noqa: PLC0415

    paths = plot_all(summary, out)
    for name, (png, svg) in paths.items():
        console.print(f"[green]{name}[/]: {png.name} + {svg.name}")


def main() -> None:  # pragma: no cover - thin wrapper
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
