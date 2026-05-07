"""Embedding encoders + corpus encoding helpers."""

from vdbbench.embed.encoder import (
    EncodedBundle,
    Encoder,
    FakeEncoder,
    detect_device,
    encode_corpus,
    load_encoded_bundle,
)
from vdbbench.embed.registry import (
    DEFAULT_ENCODER_KEY,
    ENCODER_REGISTRY,
    EncoderSpec,
    build_encoder,
)
from vdbbench.embed.sentence_transformer import SentenceTransformerEncoder

__all__ = [
    "DEFAULT_ENCODER_KEY",
    "ENCODER_REGISTRY",
    "EncodedBundle",
    "Encoder",
    "EncoderSpec",
    "FakeEncoder",
    "SentenceTransformerEncoder",
    "build_encoder",
    "detect_device",
    "encode_corpus",
    "load_encoded_bundle",
]
