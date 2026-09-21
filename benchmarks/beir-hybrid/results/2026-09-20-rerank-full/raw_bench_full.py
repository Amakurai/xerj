"""FULL controlled run: nDCG@10 + calibration of the fixed-shape raw-API bench.

Per dataset: BM25 shortlists from the node (deterministic, scored once), Jev
fixed-shape arm 3x with shuffled query order (medians, repeat variance,
pairwise determinism deltas), (prob, label) pairs for a 10-bin reliability
curve of the Jev probabilities themselves. Scoring identical to main's eval.py.
Usage: python3 raw_bench_full.py <NQUERIES> <dataset>
"""
import json, math, sys, time, random, urllib.request, collections, os

U = os.environ.get("XERJ_URL", "http://localhost:9410")
KEY = open("/root/.config/xerj-bench/typesafe-api-key").read().strip()
API = "https://api.typesafe.ai/v1/systemone"
N = int(sys.argv[1]); IDX = sys.argv[2]
MODEL = "jev-1.13.0"; REPEATS = 3; BINS = 10
random.seed(20260920)

def node_search(body):
    r = urllib.request.Request(U + f"/{IDX}/_search", data=json.dumps(body).encode(),
                               method="POST", headers={"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=120).read())

def jev_fixed(query, docs):
    questions = {}
    for i, (t, x) in docs.items():
        questions[f"d{i}"] = {"type": "noul", "instructions": {
            "question": "Does `document` contain information that answers the query in the state? Judge only whether `document` is relevant to the query, not whether it is well written.",
            "document": (t + ". " + x)[:1400]}}
    body = {"state": query, "model": MODEL, "questions": questions}
    r = urllib.request.Request(API, data=json.dumps(body).encode(), method="POST",
                               headers={"content-type": "application/json", "authorization": f"Bearer {KEY}"})
    try:
        return json.loads(urllib.request.urlopen(r, timeout=180).read())
    except urllib.error.HTTPError as e:
        return {"_err": e.code, "body": e.read().decode()[:300]}

queries = {}
for l in open(f"{IDX}/queries.jsonl"):
    d = json.loads(l); queries[d["_id"]] = d["text"]
qrels = collections.defaultdict(dict)
for i, l in enumerate(open(f"{IDX}/qrels/test.tsv")):
    if i == 0: continue
    q, d, s = l.split("\t"); qrels[q][d] = int(s)

def ndcg10(ranked, rel):
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0

pilot = list(qrels.items())[:N]
shorts = {}
for qid, _ in pilot:
    res = node_search({"query": {"multi_match": {"query": queries[qid], "fields": ["title", "text"]}},
                       "size": 30, "_source": ["title", "text"]})
    shorts[qid] = [(h["_id"], h["_source"].get("title", ""), h["_source"].get("text", "")) for h in res["hits"]["hits"]]
empty = [q for q in shorts if not shorts[q]]
bm_scores = {qid: ndcg10([d[0] for d in shorts[qid]], rel) for qid, rel in pilot}
print(f"{IDX}: {len(pilot)} queries ({len(empty)} empty BM25 shortlist), {sum(len(v) for v in shorts.values())} candidate docs", flush=True)

all_runs = []           # per repeat: {qid: ndcg}
probs_runs = []         # per repeat: {qid: {docid: prob}}
pairs = []              # (prob, label) for calibration
t_tok = [0, 0]; fails = 0; walls = []
t00 = time.time()
for rep in range(REPEATS):
    order = pilot[:]; random.shuffle(order)
    run = {}; prun = {}
    for n, (qid, rel) in enumerate(order):
        if not shorts[qid]: run[qid] = 0.0; prun[qid] = {}; continue
        docs = {i: (t, x) for i, (idd, t, x) in enumerate(shorts[qid])}
        t0 = time.time(); r = jev_fixed(queries[qid], docs); walls.append(time.time() - t0)
        if "_err" in r or "answers" not in r:
            fails += 1; run[qid] = 0.0; prun[qid] = {}; print(f"  FAIL {qid}: {str(r)[:150]}", flush=True); continue
        probs = {shorts[qid][int(k[1:])][0]: v.get("noul") for k, v in r["answers"].items() if isinstance(v, dict) and v.get("noul") is not None}
        pos = {int(k[1:]): v.get("noul") for k, v in r["answers"].items() if isinstance(v, dict) and v.get("noul") is not None}
        order_i = sorted(pos.keys(), key=lambda i: (-pos[i], i))
        ranked = [shorts[qid][i][0] for i in order_i] + [shorts[qid][i][0] for i in range(len(shorts[qid])) if i not in pos]
        run[qid] = ndcg10(ranked, rel); prun[qid] = probs
        if rep == 0:
            for docid, p in probs.items(): pairs.append((p, 1 if rel.get(docid, 0) >= 1 else 0))
        u = r.get("usage") or {}; t_tok[0] += u.get("input_tokens") or 0; t_tok[1] += u.get("output_tokens") or 0
    all_runs.append(run); probs_runs.append(prun)
    mb = sum(bm_scores[q] for q in run) / len(run); mj = sum(run.values()) / len(run)
    print(f"  repeat {rep+1}: bm25={mb:.4f} jev={mj:.4f}  ({time.time()-t00:.0f}s)", flush=True)

# per-query median of the Jev arm
med = {}
for qid, _ in pilot:
    v = sorted(r[qid] for r in all_runs); med[qid] = v[len(v) // 2]
MB = sum(bm_scores.values()) / len(bm_scores)
MJ = sum(med.values()) / len(med)
wins = sum(1 for q in med if med[q] > bm_scores[q]); losses = sum(1 for q in med if med[q] < bm_scores[q])
# determinism: pairwise |delta| across repeats, first 10 non-empty queries
dmax = 0; dtot = 0; dn = 0
probe = [q for q in shorts if shorts[q]][:10]
for q in probe:
    for a in range(REPEATS):
        for b in range(a + 1, REPEATS):
            common = set(probs_runs[a][q]) & set(probs_runs[b][q])
            for k in common:
                d = abs(probs_runs[a][q][k] - probs_runs[b][q][k]); dmax = max(dmax, d); dtot += d; dn += 1
# calibration: 10 equal-width bins
bins = collections.defaultdict(lambda: [0.0, 0, 0])   # sum_conf, n, n_rel
for p, y in pairs:
    b_ = min(int(p * BINS), BINS - 1); bins[b_][0] += p; bins[b_][1] += 1; bins[b_][2] += y
ece = sum((n / len(pairs)) * abs((r_ / n) - (c / n)) for c, n, r_ in bins.values() if n)
walls.sort()
out = {"dataset": IDX, "queries": len(pilot), "empty_bm25": len(empty), "fails": fails,
       "bm25_ndcg10": round(MB, 4), "jev_fixed_median_ndcg10": round(MJ, 4),
       "wins": wins, "losses": losses, "ties": len(med) - wins - losses,
       "input_tokens": t_tok[0], "output_tokens": t_tok[1],
       "cost_usd": round(t_tok[0] * 0.042 / 1e6, 4),
       "wall_s_p50": round(walls[len(walls) // 2], 2), "wall_s_p95": round(walls[int(len(walls) * .95)], 2),
       "total_wall_min": round((time.time() - t00) / 60, 1),
       "determinism": {"pairs_compared": dn, "max_abs_delta": round(dmax, 4), "mean_abs_delta": round(dtot / max(dn, 1), 4)},
       "calibration": {"n_pairs": len(pairs), "bins": BINS, "ece": round(ece, 4),
                       "curve": {str(b_): {"mean_conf": round(c / n, 4) if n else None,
                                            "acc": round(r_ / n, 4) if n else None, "n": n}
                                  for b_, (c, n, r_) in sorted(bins.items())}},
       "per_query": {qid: {"bm25": round(bm_scores[qid], 4), "jev_median": round(med[qid], 4)} for qid, _ in pilot}}
json.dump(out, open(f"{IDX}-full-run.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "per_query"}, indent=1))
