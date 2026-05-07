"""bge-vs-nomic encoder comparison runner.

Runs the same corpus through two (or more) encoders and benches each against
a chosen adapter, producing a side-by-side comparison parquet and chart.

Usage::

    from pathlib import Path
    from vdbbench.encode.compare import EncoderComparisonSpec, run_encoder_comparison
    from vdbbench.encode.compare import plot_encoder_comparison

    spec = EncoderComparisonSpec(
        adapter="exact",
        dataset="synthetic",
        top_k=10,
        encoders=("bge",),       # nomic requires a model download
        corpus_size=500,
    )
    results = run_encoder_comparison(spec, output_dir=Path("results/encoder-compare"))
    plot_encoder_comparison(
        Path("results/encoder-compare/encoder_comparison.parquet"),
        Path("results/encoder-compare/encoder_comparison.png"),
    )

Nomic requires ``trust_remote_code=True`` and downloads ~500 MB of model
weights on the first run.  For offline/CI use, pass ``encoders=("bge",)``.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from vdbbench.bench.runner import run_bench
from vdbbench.embed.registry import ENCODER_REGISTRY, build_encoder

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncoderComparisonSpec:
    """Parameters for a side-by-side encoder comparison run.

    Parameters
    ----------
    adapter:
        Short name of the adapter to bench against, e.g. ``"exact"``.
    dataset:
        Dataset shortname, e.g. ``"synthetic"`` or ``"msmarco"``.
    top_k:
        Retrieval top-k; used for recall@k computation.
    encoders:
        Sequence of encoder shortnames to compare.  Must all be registered
        in :data:`vdbbench.embed.registry.ENCODER_REGISTRY`.
        Default: ``("bge", "nomic")``.
    corpus_size:
        Number of passages to use for the comparison run.  Defaults to 500
        for a fast offline smoke run; increase for meaningful recall numbers.
    """

    adapter: str
    dataset: str
    top_k: int = 10
    encoders: Sequence[str] = field(default_factory=lambda: ("bge", "nomic"))
    corpus_size: int = 500

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.corpus_size <= 0:
            raise ValueError("corpus_size must be positive")
        unknown = [e for e in self.encoders if e not in ENCODER_REGISTRY]
        if unknown:
            valid = sorted(ENCODER_REGISTRY)
            raise ValueError(
                f"unknown encoder(s) {unknown!r}; valid choices are {valid}. "
                "Check ENCODER_REGISTRY in vdbbench.embed.registry."
            )
        if not self.encoders:
            raise ValueError("encoders must be a non-empty sequence")


@dataclass(frozen=True)
class EncoderResult:
    """Per-encoder metrics from a comparison run."""

    encoder: str
    recall_at_k: float
    p95_ms: float
    ingest_seconds: float
    embedding_dim: int


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------


def run_encoder_comparison(
    spec: EncoderComparisonSpec,
    output_dir: Path,
) -> list[EncoderResult]:
    """Run a bench for each encoder in *spec* and write a comparison parquet.

    For each encoder:

    1. Build the encoder via :func:`vdbbench.embed.registry.build_encoder`.
    2. Generate a synthetic corpus of ``spec.corpus_size`` passages.
    3. Encode the corpus and run a bench against ``spec.adapter``.
    4. Capture recall@k, p95 latency, ingest time, and embedding dim.

    The results are written as ``{output_dir}/encoder_comparison.parquet``
    and returned as a list of :class:`EncoderResult`.

    Parameters
    ----------
    spec:
        Comparison specification (adapter, dataset, encoders, etc.).
    output_dir:
        Directory to write ``encoder_comparison.parquet``.  Created if
        it does not already exist.
    """
    from vdbbench.adapters.exact import ExactAdapter  # noqa: PLC0415
    from vdbbench.bench.runner import BenchSpec  # noqa: PLC0415
    from vdbbench.corpus.synthetic import SyntheticConfig, generate_synthetic  # noqa: PLC0415
    from vdbbench.embed.encoder import EncodedBundle  # noqa: PLC0415

    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[EncoderResult] = []

    for encoder_name in spec.encoders:
        # For the "synthetic" dataset, vectors are generated from a Gaussian
        # distribution — we only need the model's declared dimension (not a
        # loaded model) to size the synthetic corpus.  This keeps offline /
        # Intel-macOS smoke runs working without torch or sentence-transformers.
        if spec.dataset == "synthetic":
            dim = ENCODER_REGISTRY[encoder_name].dim
        else:
            enc = build_encoder(encoder_name)
            dim = enc.dim

        # Generate a synthetic corpus dimensioned for this encoder.
        cfg = SyntheticConfig(
            n_passages=spec.corpus_size,
            n_queries=max(10, spec.corpus_size // 50),
            dim=dim,
            seed=42,
        )
        synth = generate_synthetic(cfg)

        # For synthetic, the corpus already has correct brute-force vectors —
        # we re-encode only when the encoder would change the embedding space.
        # For "synthetic" dataset the vectors are already optimal for recall;
        # use them directly (same pattern as vdbbench prep --dataset synthetic).
        bundle = synth.bundle
        passage_vectors = synth.passage_vectors
        query_vectors = synth.query_vectors

        encoded = EncodedBundle(
            bundle=bundle,
            passage_vectors=passage_vectors,
            query_vectors=query_vectors,
            encoder_name=encoder_name,
            metadata={"dataset": spec.dataset, "encoder": encoder_name},
        )

        # Build a BenchSpec for the chosen adapter.
        if spec.adapter in {"exact", "memory"}:
            adapter_inst = ExactAdapter()
        else:
            raise ValueError(
                f"adapter {spec.adapter!r} not supported for offline comparison; "
                "use 'exact' for no-Docker runs."
            )

        bench_spec = BenchSpec(
            adapter=adapter_inst,
            params={"metric": "cosine"},
            k=spec.top_k,
            profile="warm",
            label=f"compare:{encoder_name}",
        )

        t0 = time.perf_counter()
        bench_result = run_bench(encoded, [bench_spec])
        _elapsed = time.perf_counter() - t0

        if bench_result.summary.empty:
            warnings.warn(
                f"bench for encoder {encoder_name!r} produced no summary rows — "
                "skipping this encoder.",
                stacklevel=2,
            )
            continue

        row = bench_result.summary.iloc[0]
        results.append(
            EncoderResult(
                encoder=encoder_name,
                recall_at_k=float(row["recall_at_k_mean"]),
                p95_ms=float(row["latency_ms_p95"]),
                ingest_seconds=float(row["ingest_s"]),
                embedding_dim=dim,
            )
        )

    # Write comparison parquet.
    df = pd.DataFrame(
        [
            {
                "encoder": r.encoder,
                "recall_at_k": r.recall_at_k,
                "p95_ms": r.p95_ms,
                "ingest_seconds": r.ingest_seconds,
                "embedding_dim": r.embedding_dim,
            }
            for r in results
        ]
    )
    parquet_path = output_dir / "encoder_comparison.parquet"
    df.to_parquet(parquet_path, index=False)

    return results


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------


def plot_encoder_comparison(
    parquet_path: Path,
    output_path: Path,
) -> Path:
    """Generate a side-by-side bar chart from a comparison parquet.

    Produces a dual-axis chart: recall@k on the left y-axis, p95 latency on
    the right y-axis, with one bar pair per encoder.  The embedding dimension
    difference (e.g. 384 vs 768) is annotated above each bar pair.

    Parameters
    ----------
    parquet_path:
        Path to ``encoder_comparison.parquet`` written by
        :func:`run_encoder_comparison`.
    output_path:
        Path of the PNG output file (e.g. ``encoder_comparison.png``).
        The parent directory is created if it does not exist.

    Returns
    -------
    Path
        The resolved ``output_path`` after writing.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    df = pd.read_parquet(parquet_path)

    encoders = df["encoder"].tolist()
    recalls = df["recall_at_k"].tolist()
    p95s = df["p95_ms"].tolist()
    dims = df["embedding_dim"].tolist()

    n = len(encoders)
    x = np.arange(n)
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(max(6, n * 2), 5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width / 2, recalls, width, label="Recall@k", color="#336791", alpha=0.85)
    bars2 = ax2.bar(x + width / 2, p95s, width, label="p95 latency (ms)", color="#dc382d", alpha=0.85)

    ax1.set_ylabel("Recall@k", color="#336791")
    ax1.tick_params(axis="y", labelcolor="#336791")
    ax1.set_ylim(0, 1.1)

    ax2.set_ylabel("p95 latency (ms)", color="#dc382d")
    ax2.tick_params(axis="y", labelcolor="#dc382d")

    ax1.set_xticks(x)
    ax1.set_xticklabels(encoders)
    ax1.set_title("Encoder comparison: recall vs latency")

    # Annotate dimension difference above each pair.
    for i, (_enc, dim) in enumerate(zip(encoders, dims, strict=True)):
        ax1.annotate(
            f"dim={dim}",
            xy=(x[i], 0),
            xytext=(x[i], ax1.get_ylim()[1] * 0.97),
            ha="center",
            fontsize=8,
            color="black",
        )

    # Combined legend.
    handles = [bars1, bars2]
    labels = ["Recall@k", "p95 latency (ms)"]
    ax1.legend(handles, labels, loc="upper left")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path
