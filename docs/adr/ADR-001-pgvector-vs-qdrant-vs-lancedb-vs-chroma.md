# ADR-001: Why pgvector, Qdrant, LanceDB, and Chroma — and not the others

## Status

Accepted.

## Context

The benchmark has to pick four representative vector stores. The space is
big: pgvector, pgvector-rs, Qdrant, Weaviate, Milvus, Pinecone, Vespa,
Vald, Marqo, LanceDB, Chroma, Faiss, hnswlib, USearch, txtai, Elasticsearch
(`dense_vector`), OpenSearch k-NN, Redis, Mongo Atlas Vector, Supabase
(pgvector), pgvecto.rs, etc. Trying to bench all of them produces a
thin comparison full of footnotes; picking four lets every adapter get the
care it needs (correct knob plumbing, deterministic teardown, real
methodology).

## Decision

Pick four representatives along two axes — *deployment shape* (server vs
embedded) and *tuning surface* (what knobs the engine exposes):

| Adapter  | Shape          | Tuning surface                     | Why this one |
| -------- | -------------- | ---------------------------------- | ------------ |
| pgvector | server (Postgres) | HNSW (m, ef_construction, ef_search), IVFFLAT (lists, probes) | The "you already have Postgres" path. Real-world deployments lean on it heavily. |
| Qdrant   | server (Rust)  | HNSW (m, ef_construct, ef_search), payload indexes | The "purpose-built standalone" path. Strong knob discoverability. |
| LanceDB  | embedded       | IVF-PQ (num_partitions, num_sub_vectors, nprobes) | The "no service, columnar storage" path. Different index family (IVF-PQ vs HNSW). |
| Chroma   | embedded       | none (engine picks)                | The "no-tuning baseline". Important to keep so we can answer "what does default-ANN look like?" |

## Consequences

* The matrix covers two index families (HNSW + IVF-PQ) and three
  deployment shapes (Postgres, standalone server, embedded), which is
  enough to surface the three interesting tradeoffs the README claims to
  measure: knob sensitivity, throughput, memory footprint.
* Excluded: Weaviate / Milvus / Vespa (operationally heavy, dominated by
  Qdrant in the "purpose-built" slot for this comparison), Pinecone /
  Mongo Atlas Vector (hosted only, not reproducible), Faiss / hnswlib
  (libraries, not stores — the comparison is unfair). Worth revisiting
  when the bench has a "library-as-adapter" mode.
* Adding a fifth adapter has a real cost: knob plumbing, teardown
  invariants, integration container, contract test coverage. The bar for
  a new adapter is "what does this surface that the existing four
  don't?" — not "it's popular".
