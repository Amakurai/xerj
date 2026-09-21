# Vector quantization: int8 scoring at recall@10 ≈ 0.998

> **Read this first.** `scalar8` scores come from 1-byte-per-dimension
> codes written at INGEST time into a flat, slot-addressed u8 array
> ([#392](https://github.com/xerj-org/xerj/issues/392)) — the kNN serving
> working set is those codes (1 byte/dim), not the f32 `_source` vectors
> (4 bytes/dim), and the score a document gets is a function of the index
> state alone, not of the filter or candidate set of the query that found it.
> `_source` keeps the original f32 vectors, so this shrinks the scoring
> working set ~4×, not total process memory. When the code store cannot
> serve (right after open, mid-rebuild, or after a wrong-dimension write
> breaks coverage), queries fall back to the exact `_source` scan.

## The problem

Dense vectors are heavy. A 768-dim `float32` embedding is ~3 KB; a million
of them is ~3 GB of vector data that has to be resident to serve low-latency
kNN. Scale to tens of millions and the vector working set — not the text,
not the postings — becomes the thing that decides how much RAM you rent.

The standard fix is **scalar quantization**: store each dimension in one
byte instead of four. The catch everyone worries about is recall — does
compressing the vectors quietly wreck ranking quality? That is the question
this recipe answers, and the answer is no.

## Why XERJ

XERJ lets you opt a `dense_vector` field into **scalar8** (int8)
quantization per field. When you do, the kNN *serving* path scores against
1-byte-per-dimension codes instead of 4-byte floats, while `_source` still
returns the **original** vectors for retrieval. It's off by default (full
float32), so you choose the precision model per field, spelled exactly like
Elasticsearch's `int8_hnsw`.

On a real 128-dim corpus the recall cost is negligible: **recall@10 = 0.998**
against the exact float32 index. That number is computed by the run below,
not stipulated.

The codes are 128 bytes rather than 512 per vector, and since #392 they are
held resident by the engine: the run reads the live array's byte size
straight off the server (`GET /{index}/_stats` →
`primaries.sq8.fields.<field>.codes_bytes`) and checks it is exactly
`docs × dim`. The kNN scan scores against that array without touching the f32
vectors — the ~4× smaller vector working set is a property of the serving
path now, measured, not a ratio the encoding merely implies.

## The solution

Opt a field in at mapping time with `index_options.type: int8_hnsw`:

```bash
curl -sX PUT "$XERJ_URL/docs" -H 'content-type: application/json' -d '{
  "mappings": {
    "properties": {
      "title": { "type": "text" },
      "v": {
        "type": "dense_vector",
        "dims": 128,
        "similarity": "cosine",
        "index_options": { "type": "int8_hnsw" }
      }
    }
  }
}'
```

Index and query exactly as you would a full-precision field — nothing else
changes:

```bash
curl -sX POST "$XERJ_URL/docs/_search" -H 'content-type: application/json' -d '{
  "knn": { "field": "v", "query_vector": [0.12, 0.08, -0.31, "..."], "k": 10 }
}'
```

The scores come back slightly different from an exact float32 index (that's
the quantization at work — a query that exactly matches a stored vector
scores ~0.99999 instead of 1.0), but the **ranking is the same**.

## Try it

`docs/examples/vector-quantization/quant_demo.py` (the mirrored
`recipes/vector_quantization.py` runs the same demo) embeds the 40 real KB
articles into 128-dim vectors, indexes the same vectors into a float32 index
and a scalar8 index, and prints the side-by-side top hits, the measured
recall@10, and the measured byte footprint of each encoding:

```
$ python3 docs/examples/vector-quantization/quant_demo.py
embedded 40 real KB articles into 128-dim vectors

indexed into `vq-none` (float32) and `vq-scalar8` (int8_hnsw / scalar8)

query: 'how do I stop an agent's context window from overflowing?'

── float32 (exact)
    0.67958  Long-context windows do not replace memory
    0.60029  p95 latency budgets for interactive RAG agents
    0.59712  SOC 2 controls that apply to vector workloads

── scalar8 (quantized)
    0.67938  Long-context windows do not replace memory
    0.59983  p95 latency budgets for interactive RAG agents
    0.59766  SOC 2 controls that apply to vector workloads

recall@10 (scalar8 vs float32 ground truth): 0.998
resident SQ8 codes (from _stats, 40 live docs): 5120 B (128 B/vec) — serving=True
encoding size over 40 vecs: float32 = 20480 B (512 B/vec)  →  scalar8 = 5120 B (128 B/vec)  (4.00x smaller)

OK — recall preserved through 1-byte-per-dim codes, served from the
ingest-time slot-addressed array (#392). `_source` still holds the originals.
```

Both size lines are real measurements. The client-side one encodes every
corpus vector as float32 bytes (`struct`) and as int8 codes and compares the
byte totals — 20480 B vs 5120 B, exactly 4.00×. The `resident SQ8 codes` line
is read off the server (`_stats` → `primaries.sq8.fields.v.codes_bytes`) and
matches it byte-for-byte: 40 docs × 128 dim = 5120 B, `serving=True` — the
array the kNN scan actually scores against ([#392](https://github.com/xerj-org/xerj/issues/392)).

## Reproduce it yourself

```bash
# 1. Start XERJ (dev mode, default ES-compat port 9200)
xerj --insecure --data-dir ./data &

# 2. Run the demo (stdlib-only Python 3, no packages, no API keys)
python3 docs/examples/vector-quantization/quant_demo.py
```

`XERJ_URL` overrides the server (default `http://localhost:9200`); `XERJ_KB`
overrides the KB path (default: auto-discovered `demo/data/ai_kb.ndjson`).
The embedder and corpus are deterministic, so a customer should see exactly:

- `recall@10 (scalar8 vs float32 ground truth): 0.998`
- `resident SQ8 codes (from _stats, 40 live docs): 5120 B (128 B/vec) — serving=True`
- `encoding size over 40 vecs: float32 = 20480 B (512 B/vec)  →  scalar8 = 5120 B (128 B/vec)  (4.00x smaller)`

These numbers are stable run-to-run (verified across repeated runs — no
variance); the printed kNN scores are likewise identical each run.

## Notes and limits

- **Opt-in per field.** Fields without `int8_hnsw` keep exact float32
  scoring, byte-for-byte unchanged.
- **`_source` is never quantized.** Retrieval returns the vectors you
  indexed; only the scoring path uses the compact codes.
- **`scalar8` is wired; `binary` is not yet.** Binary (1-bit) quantization
  is rejected at startup rather than silently storing full precision.
- **Cosine is normalised** before quantizing for the tightest code range;
  `dot_product` and `l2_norm` similarities are supported too.
- **The serving working set shrinks ~4×; total memory does not.** The kNN
  scan scores against the ingest-time code array (1 byte/dim) instead of the
  f32 `_source` vectors (4 bytes/dim), but `_source` keeps the originals for
  retrieval, so the process's total footprint does not fall 4×. The array
  itself is observable: `GET /{index}/_stats` → `primaries.sq8` (per field:
  `dim`, `live`, `expected`, `covered`, `ready`, `serving`, `refits`,
  `codes_bytes`).
- **Codes are written at ingest and rewritten on update.** One slot per
  document, addressed `slot * dim`, tombstoned on delete — the same slot
  discipline the HNSW slab uses. A vector that lands outside the fitted
  per-dimension range WIDENS the codebook and re-encodes every live code
  (decode → encode, each value moves by at most the new quantization step);
  nothing is ever clamped, which is the defect class
  [#371](https://github.com/xerj-org/xerj/issues/371) fixed by deleting the
  old write-once caches.
- **`_score` no longer depends on the filter.** The codebook is fitted from
  the ingested vectors and lives with the data, so the same document carries
  the same `_score` whether or not a filter removed other documents — the
  property Elasticsearch has and per-query fitting lacked. Pinned over HTTP in
  `engine/crates/xerj-api/tests/sq8_codes_are_ingest_time_and_filter_independent.rs`
  (drift is exactly 0, not merely small).
- **Fallback windows keep the exact scan.** Until the open-time walk converges
  (a restart re-derives the store in the background — WAL replay never
  re-runs vector indexing), while a publication race is possible, or after a
  wrong-dimension write breaks the store's coverage gate, queries are served
  by the exact `_source` scan. In that fallback the SQ8 branch still fits its
  codebook per query over the candidates being scored, so scores there move
  by up to SQ8's own quantization step (1/255 of the fitted per-dimension
  range) when the candidate set changes — measured on the pre-#392 path:
  max |Δ_score| 1.976e-05 with order swaps at 19/30 positions when an
  unrelated-only filter was added. That fallback is correct, just not
  filter-independent; `primaries.sq8.fields.<f>.serving` tells you which
  path you are on.
- **`scalar8` still disables ANN.** A quantized field remains excluded from
  HNSW-served kNN (the graph scores f32) and takes the codes scan above for
  unfiltered queries or the exact brute-force scan otherwise. Teaching the
  graph to traverse against the codes is the remaining open half of
  [#392](https://github.com/xerj-org/xerj/issues/392).
