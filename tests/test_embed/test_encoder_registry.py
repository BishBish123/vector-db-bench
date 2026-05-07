"""Tests for ENCODER_REGISTRY and build_encoder().

The nomic model (~500 MB) is never actually downloaded here — the
SentenceTransformer constructor is mocked to return a lightweight fake
so the registry wiring can be exercised without any model I/O.

Mocking strategy
----------------
On Intel macOS ``sentence_transformers`` is not installed (no torch wheel).
We inject a fake ``sentence_transformers`` module into ``sys.modules`` via
``patch.dict`` before calling ``build_encoder()``.  Inside
``SentenceTransformerEncoder.__init__`` the ``from sentence_transformers
import SentenceTransformer`` resolves from ``sys.modules``, so it picks up
our fake class.  This is the standard approach for mocking an optional
dependency that may not be installed.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from vdbbench.embed.registry import (
    DEFAULT_ENCODER_KEY,
    ENCODER_REGISTRY,
    EncoderSpec,
    build_encoder,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_st_instance(dim: int) -> MagicMock:
    """Return a MagicMock that looks like a loaded SentenceTransformer."""
    fake = MagicMock()
    fake.get_sentence_embedding_dimension.return_value = dim
    return fake


def _make_fake_st_module(instance: MagicMock) -> tuple[ModuleType, MagicMock]:
    """Return (fake_module, mock_class) that together mock sentence_transformers.

    The returned mock_class, when called, returns ``instance``.
    ``fake_module.SentenceTransformer`` is the same mock_class so
    ``from sentence_transformers import SentenceTransformer`` binds to it.
    """
    mock_cls = MagicMock(return_value=instance)
    mod = ModuleType("sentence_transformers")
    mod.SentenceTransformer = mock_cls  # type: ignore[attr-defined]
    return mod, mock_cls


# ---------------------------------------------------------------------------
# Registry shape — no mocking needed, pure data checks
# ---------------------------------------------------------------------------


class TestEncoderRegistry:
    def test_both_encoders_present(self) -> None:
        assert "bge" in ENCODER_REGISTRY
        assert "nomic" in ENCODER_REGISTRY

    def test_bge_spec(self) -> None:
        spec = ENCODER_REGISTRY["bge"]
        assert isinstance(spec, EncoderSpec)
        assert spec.hf_id == "BAAI/bge-small-en-v1.5"
        assert spec.dim == 384
        assert spec.trust_remote_code is False

    def test_nomic_spec(self) -> None:
        spec = ENCODER_REGISTRY["nomic"]
        assert isinstance(spec, EncoderSpec)
        assert spec.hf_id == "nomic-ai/nomic-embed-text-v1.5"
        assert spec.dim == 768
        assert spec.trust_remote_code is True

    def test_default_key_is_bge(self) -> None:
        assert DEFAULT_ENCODER_KEY == "bge"
        assert DEFAULT_ENCODER_KEY in ENCODER_REGISTRY

    def test_specs_are_frozen(self) -> None:
        """EncoderSpec is a frozen dataclass — mutation must raise."""
        spec = ENCODER_REGISTRY["bge"]
        with pytest.raises((AttributeError, TypeError)):
            spec.dim = 999  # type: ignore[misc]

    def test_dims_differ_between_encoders(self) -> None:
        """bge and nomic must have different dims — the whole point of the demo."""
        assert ENCODER_REGISTRY["bge"].dim != ENCODER_REGISTRY["nomic"].dim


# ---------------------------------------------------------------------------
# build_encoder() — mocked to avoid any model download
# ---------------------------------------------------------------------------


class TestBuildEncoder:
    def test_build_bge_returns_correct_dim(self) -> None:
        fake_inst = _make_fake_st_instance(384)
        fake_mod, _ = _make_fake_st_module(fake_inst)
        with patch.dict(sys.modules, {"sentence_transformers": fake_mod}):
            enc = build_encoder("bge")
        assert enc.dim == 384

    def test_build_nomic_returns_correct_dim(self) -> None:
        fake_inst = _make_fake_st_instance(768)
        fake_mod, _ = _make_fake_st_module(fake_inst)
        with patch.dict(sys.modules, {"sentence_transformers": fake_mod}):
            enc = build_encoder("nomic")
        assert enc.dim == 768

    def test_build_nomic_passes_trust_remote_code(self) -> None:
        """build_encoder('nomic') must forward trust_remote_code=True to SBERT."""
        fake_inst = _make_fake_st_instance(768)
        fake_mod, mock_cls = _make_fake_st_module(fake_inst)
        with patch.dict(sys.modules, {"sentence_transformers": fake_mod}):
            build_encoder("nomic")
        _args, kwargs = mock_cls.call_args
        assert kwargs.get("trust_remote_code") is True

    def test_build_bge_does_not_set_trust_remote_code(self) -> None:
        """bge must be loaded with trust_remote_code=False (the safe default)."""
        fake_inst = _make_fake_st_instance(384)
        fake_mod, mock_cls = _make_fake_st_module(fake_inst)
        with patch.dict(sys.modules, {"sentence_transformers": fake_mod}):
            build_encoder("bge")
        _args, kwargs = mock_cls.call_args
        assert kwargs.get("trust_remote_code") is False

    def test_default_encoder_is_bge(self) -> None:
        """Calling build_encoder() without a key must select bge (dim=384)."""
        fake_inst = _make_fake_st_instance(384)
        fake_mod, _ = _make_fake_st_module(fake_inst)
        with patch.dict(sys.modules, {"sentence_transformers": fake_mod}):
            enc = build_encoder()
        assert enc.dim == 384

    def test_none_key_uses_default(self) -> None:
        """build_encoder(None) must behave identically to build_encoder()."""
        fake_inst = _make_fake_st_instance(384)
        fake_mod, _ = _make_fake_st_module(fake_inst)
        with patch.dict(sys.modules, {"sentence_transformers": fake_mod}):
            enc = build_encoder(None)
        assert enc.dim == 384

    def test_invalid_key_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unknown encoder"):
            build_encoder("not-a-real-encoder")

    def test_invalid_key_error_lists_valid_choices(self) -> None:
        with pytest.raises(ValueError, match="bge"):
            build_encoder("typo")

    def test_encoder_name_matches_hf_id(self) -> None:
        """The encoder's .name property must equal the HuggingFace model ID."""
        fake_inst = _make_fake_st_instance(384)
        fake_mod, _ = _make_fake_st_module(fake_inst)
        with patch.dict(sys.modules, {"sentence_transformers": fake_mod}):
            enc = build_encoder("bge")
        assert enc.name == ENCODER_REGISTRY["bge"].hf_id
