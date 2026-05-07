"""Encoder registry — maps short names to HuggingFace model IDs and metadata.

Both encoders are loadable via ``sentence-transformers``.  The nomic model
requires ``trust_remote_code=True`` because it ships a custom
``modeling_hf_nomic_bert.py`` file that must be executed on load.  Read the
model card at https://huggingface.co/nomic-ai/nomic-embed-text-v1.5 before
enabling this in a security-sensitive environment.

Usage
-----
    from vdbbench.embed.registry import ENCODER_REGISTRY, build_encoder

    enc = build_encoder("bge")        # default 384-dim encoder
    enc = build_encoder("nomic")      # 768-dim encoder
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vdbbench.embed.sentence_transformer import SentenceTransformerEncoder


@dataclass(frozen=True)
class EncoderSpec:
    """Static description of a registered embedding model."""

    # Short human-readable key used on the CLI (e.g. "bge", "nomic").
    key: str
    # HuggingFace model repository ID, passed straight to SentenceTransformer.
    hf_id: str
    # Native embedding dimension at the model's default output size.
    dim: int
    # Whether SentenceTransformer must be called with trust_remote_code=True.
    # Nomic ships a custom modeling file (modeling_hf_nomic_bert.py) that
    # sentence-transformers must execute from the HuggingFace cache.  Read the
    # model card before enabling this in a security-sensitive context.
    trust_remote_code: bool = False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ENCODER_REGISTRY: dict[str, EncoderSpec] = {
    "bge": EncoderSpec(
        key="bge",
        hf_id="BAAI/bge-small-en-v1.5",
        dim=384,
        trust_remote_code=False,
    ),
    "nomic": EncoderSpec(
        key="nomic",
        hf_id="nomic-ai/nomic-embed-text-v1.5",
        dim=768,
        trust_remote_code=True,
    ),
}

#: The key used when no ``--encoder`` flag is specified.
DEFAULT_ENCODER_KEY: str = "bge"


def build_encoder(key: str | None = None) -> SentenceTransformerEncoder:
    """Return a ``SentenceTransformerEncoder`` for the named registry entry.

    Parameters
    ----------
    key:
        One of the keys in ``ENCODER_REGISTRY`` (e.g. ``"bge"`` or
        ``"nomic"``).  ``None`` selects ``DEFAULT_ENCODER_KEY``.

    Raises
    ------
    ValueError
        If ``key`` is not present in ``ENCODER_REGISTRY``.
    ImportError
        If ``sentence-transformers`` is not installed.
    """
    resolved = key or DEFAULT_ENCODER_KEY
    if resolved not in ENCODER_REGISTRY:
        valid = sorted(ENCODER_REGISTRY)
        raise ValueError(
            f"unknown encoder {resolved!r}; valid choices are {valid}. "
            "Check ENCODER_REGISTRY in vdbbench.embed.registry."
        )
    spec = ENCODER_REGISTRY[resolved]
    # Import here so the registry module stays importable without torch.
    from vdbbench.embed.sentence_transformer import (  # noqa: PLC0415
        SentenceTransformerEncoder,
    )

    return SentenceTransformerEncoder(
        model_name=spec.hf_id,
        trust_remote_code=spec.trust_remote_code,
    )
