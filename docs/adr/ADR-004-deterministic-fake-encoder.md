# ADR-004: Why ship a deterministic FakeEncoder

## Status

Accepted.

## Context

The default real encoder is `BAAI/bge-small-en-v1.5` via
sentence-transformers. That carries three costs that bite tests and CI:

1. **Size.** ~130 MB of model weights downloaded on first use, cached in
   `~/.cache/huggingface`.
2. **Platform gating.** `sentence-transformers` requires `torch`, which
   no longer ships macOS x86_64 wheels at version ≥ 2.3. CI on Intel
   macOS would fail.
3. **Determinism.** Even with the same model weights, encoder output can
   vary by 1–2 ulps across torch versions and devices (CPU vs MPS vs
   CUDA). Test assertions like "this exact pid retrieves first" need
   bit-stable embeddings.

A deterministic, in-package encoder dodges all three.

## Decision

Ship `vdbbench.embed.encoder.FakeEncoder` as a first-class encoder, not
just a test fixture. Its contract:

* Hash-based: `vec[i] = blake2b(text[i])[:dim]` interpreted as float32,
  then L2-normalized.
* Deterministic across processes, OSes, torch versions, hardware.
* Cap at `dim = 64` because the hash output is the source of entropy.
  Higher dims would have to recycle bytes and stop being well-distributed.
* Same `Encoder` Protocol as the real one, so any code that takes an
  `Encoder` works with both.

The encoder is exported from `vdbbench.embed` so users can import it
without poking at the test tree.

## Use cases

`FakeEncoder` is exercised by `tests/test_embed/test_encoder.py` and the
smoke pipeline script `scripts/smoke_pipeline.py` (which now drives a
full `corpus → encode → save → load → bench → plot` round-trip against
an in-memory adapter, in seconds, with no torch and no network).

* **Unit tests.** Every test in `tests/test_embed`, `tests/test_bench`,
  and the adapter contract test uses it. No model download, no torch
  dependency, no network.
* **Smoke runs in CI.** `scripts/smoke_pipeline.py` runs end-to-end
  against an in-memory adapter using only `FakeEncoder`, in seconds.
* **Adapter regression spotting.** Because the encoder is text-deterministic,
  the gold pid for a given query is `argmax(cos(query_vec, *passage_vecs))`
  computed once and pinned. An adapter that mis-orders results fails
  loudly.

## Consequences

* `FakeEncoder` is *not* a high-quality encoder. It produces no semantic
  signal — two paraphrases hash to unrelated vectors. The headline
  numbers in the README come from the real encoder; `FakeEncoder` is
  for correctness, not quality.
* The `dim ≤ 64` cap is documented in the constructor and tested. Pushing
  past it would silently degrade vector quality without any error.
