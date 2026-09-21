#!/usr/bin/env python3
"""Per-arm search latency, closed loop, one client.

    XERJ_API_KEY=... python3 latency.py http://localhost:10840 scifact /path/to/beir/scifact 300 [ranked.json]

With the optional last argument, every timed response's ranked `(_id, _score)`
list is written there, per arm and query, so two binaries' latency runs can be
checked for identical results with `compare_ranked.py` over EXACTLY the
requests that were timed (identity.py covers fewer queries, deeper).

Read-only (`_search`, `/v1/metrics`). Every arm sends N DISTINCT query texts,
so no request inside an arm repeats. Arms differ in request body, and the
result cache is keyed on the whole body, so arms cannot serve each other. That
is not taken on trust: the node's own result-cache hit gauge is scraped around
every arm and the delta printed. A row whose `cache_hits` is not 0 measured
the cache, not the engine.

The last arm repeats ONE request on purpose and is expected to show hits: it
is there so the trap stays visible to whoever runs this next.

Machine load (1-minute loadavg) is printed per arm. A latency taken while
other jobs saturate the box is an upper bound, not a measurement of the engine.
"""
import json
import sys
import time

from common import (bm25_query, cache_counters, distinct_queries, http, hybrid_query, loadavg, pct,
                    semantic_query)

url, index, data_dir, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
ranked_out = sys.argv[5] if len(sys.argv) > 5 else None
ranked = {}
qs = distinct_queries(data_dir, n)


def search(query, size=10):
    r = http(url, "POST", f"/{index}/_search", {"size": size, "_source": False, "query": query})
    if "_err" in r:
        raise SystemExit(f"search failed: {r}")
    return r


def timed(query):
    t = time.perf_counter()
    r = search(query)
    return time.perf_counter() - t, r


# The ids filter names the BM25 top-30 of the same text. Those lookups are
# made up front so they are not inside any timed region.
print("preparing ids filters (untimed) ...", flush=True)
top30 = {q: [h["_id"] for h in search(bm25_query(q), 30)["hits"]["hits"]] for q in qs}

TERM = {"term": {"bucket": "b3"}}
TERM_AND_RANGE = {"bool": {"filter": [{"term": {"bucket": "b3"}}, {"range": {"n": {"gte": 100, "lt": 900}}}]}}

arms = [
    ("bm25", lambda q: timed(bm25_query(q))),
    ("semantic (unfiltered)", lambda q: timed(semantic_query(q))),
    ("semantic + ids filter (30 docs)", lambda q: timed(semantic_query(q, flt={"ids": {"values": top30[q]}}))),
    ("semantic + term filter (~10%)", lambda q: timed(semantic_query(q, flt=TERM))),
    ("semantic + term AND range filter", lambda q: timed(semantic_query(q, flt=TERM_AND_RANGE))),
    ("hybrid rrf", lambda q: timed(hybrid_query(q))),
    ("semantic, SAME request repeated", lambda q: timed(semantic_query(qs[0], k=11))),
]
print(f"index={index} docs={http(url, 'GET', f'/{index}/_count').get('count')} distinct_queries={len(qs)}")
for name, fn in arms:
    hits0, _ = cache_counters(url)
    load0 = loadavg()
    print("ARM_START", name, time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()), flush=True)
    lat, took, rows = [], [], []
    for q in qs:
        dt, r = fn(q)
        lat.append(dt * 1000)
        took.append(r.get("took", 0))
        rows.append([[h["_id"], h.get("_score")] for h in r["hits"]["hits"]])
    ranked[name] = rows
    print("ARM_END", name, time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()), flush=True)
    hits1, _ = cache_counters(url)
    lat.sort()
    took.sort()
    print(f"{name:34s} n={len(lat)} p50={pct(lat, .5):7.1f}ms p95={pct(lat, .95):7.1f}ms"
          f"  server_took p50={pct(took, .5)}ms p95={pct(took, .95)}ms"
          f"  cache_hits={hits1 - hits0}  loadavg={load0}->{loadavg()}", flush=True)
if ranked_out:
    with open(ranked_out, "w") as f:
        json.dump(ranked, f)
    print("wrote", ranked_out)
