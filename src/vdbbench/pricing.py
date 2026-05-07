"""Cloud pricing table for the $/M-queries cost metric.

Each entry represents the smallest plausible production-grade tier for the
named adapter. Prices are sourced from official pricing pages; the
``verified`` flag marks whether the figure was confirmed by fetching the
live page, or is an estimate/unverified.

Formula (documented in README § Cost methodology):

    qps = total_queries / total_query_seconds
    hourly_query_capacity = qps * 3600
    cost_per_query = compute_per_hour_usd / hourly_query_capacity + per_query_usd
    cost_per_million_queries_usd = cost_per_query * 1_000_000

``compute_per_hour_usd`` is the smallest production-tier compute price.
``per_query_usd`` captures any metered per-query charge on top of compute.

For adapters where the pricing model is not hourly-compute (e.g. Chroma
Cloud charges per TiB queried, not per compute hour), the cost is expressed
as an effective ``per_query_usd`` and ``compute_per_hour_usd`` is set to 0.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdapterPricing:
    """Pricing entry for one adapter."""

    adapter: str
    compute_per_hour_usd: float
    per_query_usd: float
    tier: str
    source_url: str
    as_of: str
    verified: bool
    notes: str = ""


# ---------------------------------------------------------------------------
# Pricing table
# ---------------------------------------------------------------------------
# Prices as of 2025-05 / 2026-05 per official sources (see source_url).
# Unverified entries are marked verified=False with a note explaining why.
# ---------------------------------------------------------------------------

PRICING_TABLE: dict[str, AdapterPricing] = {
    # Neon Scale tier: $0.222/CU-hour confirmed live from neon.com/pricing
    # (fetched 2026-05-06). A CU is ~4 GB RAM + associated CPU — the Neon
    # pricing page defines 1 CU as "~4 GB RAM + CPU + SSD"; the autoscaling
    # docs confirm the ratio (e.g. 2 CU = 8 GB RAM). The earlier comment
    # "0.25 vCPU + 1 GB RAM" was incorrect. pgvector runs inside Neon.
    # Source: https://neon.com/pricing (fetched 2026-05-06).
    "pgvector": AdapterPricing(
        adapter="pgvector",
        compute_per_hour_usd=0.222,
        per_query_usd=0.0,
        tier="Neon Scale — 1 CU",
        source_url="https://neon.com/pricing",
        as_of="2026-05",
        verified=True,
        notes=(
            "$0.222/CU-hour confirmed from neon.com/pricing (fetched 2026-05-06). "
            "1 CU = ~4 GB RAM + CPU (Neon defines 1 CU as '~4 GB RAM + CPU + SSD'; "
            "autoscaling docs confirm 2 CU = 8 GB RAM). "
            "Prior comment '0.25 vCPU + 1 GB RAM' was incorrect."
        ),
    ),
    # Qdrant Cloud Standard tier: specific hourly rates are not published on
    # the public pricing page (qdrant.tech/pricing) — the page says
    # "usage-based pricing" with a calculator / talk-to-sales flow.
    # $0.08/hr is the figure from earlier public documentation and community
    # references; mark unverified because it could not be confirmed from the
    # live page on 2026-05.
    # TODO: verify at https://cloud.qdrant.io or via the Qdrant pricing calculator
    "qdrant": AdapterPricing(
        adapter="qdrant",
        compute_per_hour_usd=0.08,
        per_query_usd=0.0,
        tier="Qdrant Cloud Standard — smallest dedicated node",
        source_url="https://qdrant.tech/pricing/",
        as_of="2025-01",
        verified=False,
        notes=(
            "UNVERIFIED. The live pricing page does not publish specific hourly rates; "
            "this figure is from earlier public documentation. "
            "TODO: verify at https://cloud.qdrant.io or via the Qdrant pricing calculator."
        ),
    ),
    # Chroma Cloud uses a data-volume pricing model (not hourly compute):
    #   $0.0075 per TiB queried (confirmed from trychroma.com/pricing, 2026-05).
    # A typical 384-dim float32 query vector is 1536 bytes; at 1M queries
    # that is exactly 1.536 GB = 0.00140 TiB of query data transferred
    # (1536 x 1_000_000 / 1024^4 = 0.001397 TiB, rounded to 0.00140 TiB).
    # Effective per-query cost: $0.0075 / (1024*1024) * 1536 bytes ≈ $1.1e-8/query
    # (i.e. ~$0.011 / M-queries from the query-data charge alone).
    # Rounding to the nearest micro-dollar: $1.1e-8/query; set
    # compute_per_hour_usd=0 because Chroma Cloud is serverless.
    "chroma": AdapterPricing(
        adapter="chroma",
        compute_per_hour_usd=0.0,
        per_query_usd=1.1e-8,
        tier="Chroma Cloud — serverless",
        source_url="https://trychroma.com/pricing",
        as_of="2026-05",
        verified=True,
        notes=(
            "$0.0075/TiB-queried confirmed from trychroma.com/pricing 2026-05. "
            "Effective per-query cost assumes 384-dim float32 query vector (~1536 bytes). "
            "Serverless — no hourly compute charge."
        ),
    ),
    # LanceDB is an embedded library; the cost is purely S3 storage for the
    # index files, not compute. AWS S3 Standard is $0.023/GB-month (standard
    # first-tier, us-east-1). There is no compute or per-query charge for the
    # embedded library itself (runs in-process).
    # TODO: verify current S3 Standard rate at https://aws.amazon.com/s3/pricing/
    "lancedb": AdapterPricing(
        adapter="lancedb",
        compute_per_hour_usd=0.0,
        per_query_usd=0.0,
        tier="LanceDB embedded + S3 Standard storage",
        source_url="https://aws.amazon.com/s3/pricing/",
        as_of="2025-01",
        verified=False,
        notes=(
            "UNVERIFIED. LanceDB is self-hosted embedded; compute cost = 0. "
            "S3 Standard storage cost ($0.023/GB-month) is separate from query cost. "
            "TODO: verify current rate at https://aws.amazon.com/s3/pricing/"
        ),
    ),
    # In-process brute-force baseline — runs in the bench process itself.
    # No cloud cost; useful as a floor reference.
    "exact": AdapterPricing(
        adapter="exact",
        compute_per_hour_usd=0.0,
        per_query_usd=0.0,
        tier="In-process brute-force (no cloud)",
        source_url="",
        as_of="2026-05",
        verified=True,
        notes="In-memory baseline; no cloud compute or query charge.",
    ),
}


def cost_per_million_queries_usd(
    adapter: str,
    total_queries: int,
    total_query_seconds: float,
) -> float | None:
    """Compute estimated $/M-queries for *adapter* given measured throughput.

    Returns ``None`` when:
    - the adapter is not in :data:`PRICING_TABLE`
    - ``total_query_seconds`` is zero or negative (can't derive QPS)
    - ``total_queries`` is zero

    Returns ``None`` (which callers should convert to ``float("nan")`` for
    parquet) when:
    - the adapter is not in :data:`PRICING_TABLE`
    - the adapter is a non-cloud in-process baseline (``exact`` or any
      future "memory" alias) — returning ``0.0`` for these adapters is
      misleading because it reads as "cheapest cloud option" rather than
      "not applicable". Callers should render ``NaN`` or ``N/A``.
    - ``total_query_seconds`` is zero or negative (can't derive QPS)
    - ``total_queries`` is zero

    The formula:

    .. code-block:: text

        qps = total_queries / total_query_seconds
        hourly_query_capacity = qps * 3600
        cost_per_query = (
            compute_per_hour_usd / hourly_query_capacity   # compute share per query
            + per_query_usd                                 # metered per-query charge
        )
        cost_per_million = cost_per_query * 1_000_000

    When ``compute_per_hour_usd`` is 0 (e.g. serverless / embedded), only the
    ``per_query_usd`` component contributes.

    Non-cloud adapters
    ------------------
    ``exact`` (and any future in-process / "memory" alias) has no cloud
    compute or per-query charge — it is a local brute-force baseline.
    Returning ``0.0`` would rank it as the "cheapest" option in cost
    comparisons, which is semantically wrong.  ``None`` propagates to
    ``NaN`` in pandas/parquet, which plot code skips or labels "N/A".
    """
    # Non-cloud in-process adapters: cost is not applicable, not zero.
    _NON_CLOUD_ADAPTERS = {"exact", "memory"}
    if adapter in _NON_CLOUD_ADAPTERS:
        return None
    pricing = PRICING_TABLE.get(adapter)
    if pricing is None:
        return None
    if total_queries <= 0 or total_query_seconds <= 0:
        return None

    qps = total_queries / total_query_seconds
    hourly_capacity = qps * 3600.0

    compute_share = pricing.compute_per_hour_usd / hourly_capacity if hourly_capacity > 0 else 0.0

    per_query = compute_share + pricing.per_query_usd
    return per_query * 1_000_000.0
