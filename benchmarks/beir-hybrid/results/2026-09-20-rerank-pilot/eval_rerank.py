"""Rerank pilot: nDCG@10, BM25 arm vs BM25+Jev arm, first N judged SciFact queries.

Scoring is byte-identical to benchmarks/beir-hybrid/eval.py on origin/main (ndcg10,
qrels parsing, the multi_match BM25 query). Differences from that file:
- adds a "rerank" arm: BM25 shortlist judged by the live Jev endpoint via the
  node's rerank stage, model pinned to jev-1.13.0;
- sends "_source":["title"] + "fields":["_passage"] so the judge sees title +
  matching passage only (egress control, per docs/RERANK.md);
- interleaves arms per query (cache-neutral), logs a per-query JSONL line,
- repeats 3 queries afterwards and compares probabilities (determinism probe).
"""
import json, math, sys, time, urllib.request, collections, os

U = os.environ.get("XERJ_URL", "http://localhost:9410")
IDX = sys.argv[2] if len(sys.argv) > 2 else "scifact"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
MODEL = "jev-1.13.0"
LOG = open(f"{IDX}-rerank-pilot.jsonl", "w")

def post(p, b, timeout=120):
    r = urllib.request.Request(U + p, data=json.dumps(b).encode(), method="POST",
                               headers={"content-type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(r, timeout=timeout).read())
    except urllib.error.HTTPError as e:
        return {"_err": e.code, "body": e.read().decode()[:300]}

queries = {}
for l in open(IDX + "/queries.jsonl"):
    d = json.loads(l); queries[d["_id"]] = d["text"]
qrels = collections.defaultdict(dict)
for i, l in enumerate(open(IDX + "/qrels/test.tsv")):
    if i == 0: continue
    q, d, s = l.split("\t"); qrels[q][d] = int(s)

def ndcg10(ranked, rel):
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0

BM = lambda q: {"multi_match": {"query": q, "fields": ["title", "text"]}}
ids = lambda res: [h["_id"] for h in res.get("hits", {}).get("hits", [])]

def bm25(q, n=100):
    return ids(post(f"/{IDX}/_search", {"size": n, "_source": False, "query": BM(q)}))

def jev(q, w=30):
    body = {"query": BM(q), "size": w, "_source": ["title"], "fields": ["_passage"],
            "rerank": {"model": MODEL, "window": w}}
    res = post(f"/{IDX}/_search", body, timeout=180)
    rr = res.get("_rerank") or {}
    probs = {h["_id"]: h["_score"] for h in res.get("hits", {}).get("hits", [])
             if h.get("_score") is not None}
    return ids(res), rr, probs, res.get("_err")

pilot = list(qrels.items())[:N]
print(f"pilot: {len(pilot)} queries, model {MODEL}", flush=True)
rows = []
t_start = time.time()
for n, (qid, rel) in enumerate(pilot):
    q = queries[qid]
    t0 = time.time()
    base = bm25(q)                                   # arm A first
    j_ids, rr, probs, err = jev(q)                   # arm B immediately after
    wall = time.time() - t0
    row = {"qid": qid, "bm25": ndcg10(base, rel), "jev": ndcg10(j_ids, rel) if j_ids else None,
           "applied": rr.get("applied"), "judged": rr.get("judged"), "unjudged": rr.get("unjudged"),
           "partial_failures": rr.get("partial_failures"), "reason": rr.get("reason"),
           "in_tok": (rr.get("usage") or {}).get("input_tokens"),
           "out_tok": (rr.get("usage") or {}).get("output_tokens"),
           "took_ms": rr.get("took_ms"), "err": err, "probs": probs}
    rows.append(row); LOG.write(json.dumps({k: v for k, v in row.items() if k != "probs"}) + "\n"); LOG.flush()
    if n % 10 == 0: print(f"  {n+1}/{len(pilot)} ({time.time()-t_start:.0f}s)", flush=True)

ok = [r for r in rows if r["jev"] is not None]
deg = [r for r in rows if r["applied"] is False or r["err"]]
lat = sorted(r["took_ms"] for r in ok if r["took_ms"] is not None)
mb = sum(r["bm25"] for r in rows) / len(rows)
mj = sum(r["jev"] for r in ok) / len(ok)
wins = sum(1 for r in ok if r["jev"] > r["bm25"]); losses = sum(1 for r in ok if r["jev"] < r["bm25"])
ti = sum(r["in_tok"] or 0 for r in ok); to = sum(r["out_tok"] or 0 for r in ok)
judged = sum(r["judged"] or 0 for r in ok); unjudged = sum(r["unjudged"] or 0 for r in ok)
print(f"\nqueries scored: {len(ok)}  degraded/errored: {len(deg)}  judged: {judged}  unjudged: {unjudged}")
print(f"bm25   nDCG@10 = {mb:.4f}")
print(f"bm25+jev nDCG@10 = {mj:.4f}   (paired: wins {wins} / losses {losses} / ties {len(ok)-wins-losses})")
if lat: print(f"rerank took_ms p50={lat[len(lat)//2]} p95={lat[int(len(lat)*.95)]}")
print(f"usage: input {ti} tok, output {to} tok  -> ${ti*0.042/1e6:.4f} at $0.042/Mtok input-only")

# determinism probe: re-ask 3 pilot queries, compare probabilities
print("\ndeterminism probe (3 queries re-asked):")
for r in rows[:3]:
    q = queries[r["qid"]]
    _, rr2, probs2, _ = jev(q)
    common = set(r["probs"]) & set(probs2)
    if not common: print(f"  {r['qid']}: no overlap ({len(probs2)} answers)"); continue
    dmax = max(abs(r["probs"][k] - probs2[k]) for k in common)
    same = sum(1 for k in common if r["probs"][k] == probs2[k])
    print(f"  {r['qid']}: {same}/{len(common)} probs identical, max |delta| = {dmax}")
