"""CLI entry point. Subcommands fill in as later phases land."""

from __future__ import annotations

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
    sample_size: int = typer.Option(100_000, help="Number of MS-MARCO passages to sample."),
    embed_model: str = typer.Option(
        "BAAI/bge-small-en-v1.5", help="sentence-transformers model id."
    ),
    seed: int = typer.Option(42, help="Random seed for deterministic sampling."),
) -> None:
    """Build corpus + embeddings + ground-truth qrels (Phase 1, lands next commit)."""
    console.print(
        f"[yellow]prep[/] not implemented yet. "
        f"Will build sample_size={sample_size}, model={embed_model}, seed={seed}."
    )
    raise typer.Exit(code=2)


@app.command()
def bench(
    all_dbs: bool = typer.Option(False, "--all", help="Run against every implemented DB."),
) -> None:
    """Run the benchmark battery (Phase 3)."""
    console.print(f"[yellow]bench[/] not implemented yet. all={all_dbs}")
    raise typer.Exit(code=2)


@app.command()
def plot() -> None:
    """Regenerate analysis plots from results/raw.parquet (Phase 4)."""
    console.print("[yellow]plot[/] not implemented yet.")
    raise typer.Exit(code=2)


def main() -> None:  # pragma: no cover - thin wrapper
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
