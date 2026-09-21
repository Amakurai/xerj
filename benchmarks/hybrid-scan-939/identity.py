#!/usr/bin/env python3
"""Dump full responses for a fixed query set, to compare two binaries.

    XERJ_API_KEY=... python3 identity.py http://localhost:10840 scifact /path/to/beir/scifact 100 out.json

Read-only. Deeper and heavier than latency.py on purpose: `size`/`k` 100 so a
reordering far from the top still shows, and `_source` ON for the vector arms
so the hydrated documents are compared too, not only ids and scores. `took` is
dropped; everything else in the response is kept verbatim.

Run it against the OLD binary and the NEW binary over the SAME data directory
(same segments, same document order), then feed both dumps to compare.py.
"""
import json
import sys

from common import bm25_query, distinct_queries, http, hybrid_query, semantic_query

url, index, data_dir, n, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
qs = distinct_queries(data_dir, n)
K = 100


def search(query, size, source):
    r = http(url, "POST", f"/{index}/_search", {"size": size, "_source": source, "query": query})
    if "_err" in r:
        raise SystemExit(f"search failed: {r}")
    r.pop("took", None)
    return r


top30 = {q: [h["_id"] for h in search(bm25_query(q), 30, False)["hits"]["hits"]] for q in qs}
TERM = {"term": {"bucket": "b3"}}
TERM_AND_RANGE = {"bool": {"filter": [{"term": {"bucket": "b3"}}, {"range": {"n": {"gte": 100, "lt": 900}}}]}}
# Filters the projection fast path must DECLINE (not on its whitelist), so the
# fallback path is compared too.
MATCH_FILTER = {"match": {"title": "cell"}}

arms = {
    "semantic": lambda q: search(semantic_query(q, K), K, True),
    "semantic_ids": lambda q: search(semantic_query(q, K, {"ids": {"values": top30[q]}}), K, True),
    "semantic_term": lambda q: search(semantic_query(q, K, TERM), K, True),
    "semantic_term_range": lambda q: search(semantic_query(q, K, TERM_AND_RANGE), K, True),
    "semantic_match_filter": lambda q: search(semantic_query(q, K, MATCH_FILTER), K, True),
    "semantic_passage": lambda q: http(url, "POST", f"/{index}/_search", {
        "size": 10, "_source": False, "fields": ["_passage"], "query": semantic_query(q, 10)}),
    "hybrid": lambda q: search(hybrid_query(q, K), K, True),
    "bool_must_semantic_filter_ids": lambda q: search(
        {"bool": {"must": [semantic_query(q, K)], "filter": [{"ids": {"values": top30[q]}}]}}, K, False),
}
dump = {}
for name, fn in arms.items():
    rows = []
    for q in qs:
        r = fn(q)
        r.pop("took", None)
        rows.append({"q": q, "response": r})
    dump[name] = rows
    hits = sum(len(row["response"].get("hits", {}).get("hits", [])) for row in rows)
    print(f"{name:32s} queries={len(rows)} hits_total={hits}", flush=True)
with open(out, "w") as f:
    json.dump(dump, f, sort_keys=True)
print("wrote", out)
