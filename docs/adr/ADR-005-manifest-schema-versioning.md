# ADR-005: Schema-version every persisted manifest

## Status

Accepted.

## Context

Three on-disk manifests bind parquet files to the loader that produced
them:

* `corpus/manifest.json` — written by `CorpusBundle.save`, names the
  passages/queries/qrels parquet, carries the corpus fingerprint.
* `encoded/manifest.json` — written by `EncodedBundle.save`, binds
  embedding vectors to the corpus fingerprint they were produced
  against.
* `results/bench_manifest.json` — written by `run_bench`, captures
  encoder identity, adapter versions, host metadata, and bench specs.

The bench-manifest already carries `schema_version: 1`. The corpus and
encoded manifests did not, so a future format change (renamed field,
removed field, semantically different field) had no detection path —
older readers would silently mis-parse newer manifests, and newer
readers would happily attempt to read an old layout.

## Decision

Every manifest carries a `schema_version: int` field, currently `1`.
Loaders refuse to open a manifest whose version doesn't match their
expected version, raising `IncompatibleManifestError` (a `ValueError`
subclass) with both the on-disk version and the loader's expected
version in the message.

Pre-versioning manifests (no `schema_version` key) are treated as v1
because that's what they happen to be — gratuitously breaking older
on-disk corpora would lose users' work. Once we bump to v2, that
fallback becomes a real version check.

## Bumping procedure

Bump the relevant constant when the manifest changes in a way readers
care about:

* `CORPUS_MANIFEST_SCHEMA_VERSION` (in `vdbbench.corpus.bundle`)
* `ENCODED_MANIFEST_SCHEMA_VERSION` (in `vdbbench.embed.encoder`)
* `BENCH_MANIFEST_SCHEMA_VERSION` (in `vdbbench.bench.runner`)

A bump is required when:

* a field is renamed
* a field is removed
* the semantic meaning of a field changes

A bump is **not** required when:

* a new field is added that defaults sensibly when missing
* a field's documentation changes but its on-disk type/values don't

When you bump, write a one-line note in the relevant module's
`save()` docstring explaining what changed, and either keep the loader
backwards-compatible by branching on `schema_version` or document the
forced regeneration.

## Consequences

* Adding fields is cheap (no bump). Renames/removes are expensive
  (bump + branch the loader). The asymmetry is the right shape — most
  evolution is additive.
* Pre-existing on-disk bundles continue to load. The first real bump
  is when this versioning starts paying its way.
* CI gains a new pinned constant per manifest. A surprise edit that
  flips a value without a corresponding loader change shows up as a
  failing test.
