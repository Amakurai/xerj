"""Raw-API bench of the FIXED question shape (V2): document inside each noul's
instructions, state = the query string. Same 40 pilot queries, same BM25 top-30
shortlists, same scoring as eval_rerank.py. Pin jev-1.13.0. One request per
query (30 nouls), like the stage's batching.

This measures what xerj-rerank would score after build_jev_request is fixed to
the shape the TypeSafe docs themselves prescribe (per-question data fields,
referred to by name in backticks).
"""
import json, math, sys, time, urllib.request, collections, os

U = os.environ.get("XERJ_URL", "http://localhost:9410")
KEY = open("/root/.config/xerj-bench/typesafe-api-key").read().strip()
API = "https://api.typesafe.ai/v1/systemone"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
MODEL = "jev-1.13.0"

def node_search(body):
    r = urllib.request.Request(U + "/scifact/_search", data=json.dumps(body).encode(),
                               method="POST", headers={"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=120).read())

def jev_fixed(query, docs):
    """docs: list of (ordinal, title, text). One request, one noul per doc, doc inline."""
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
for l in open("scifact/queries.jsonl"):
    d = json.loads(l); queries[d["_id"]] = d["text"]
qrels = collections.defaultdict(dict)
for i, l in enumerate(open("scifact/qrels/test.tsv")):
    if i == 0: continue
    q, d, s = l.split("\t"); qrels[q][d] = int(s)

def ndcg10(ranked, rel):
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0

# corpus text for the shortlists
need = set(); shorts = {}
pilot = list(qrels.items())[:N]
for qid, _ in pilot:
    res = node_search({"query": {"multi_match": {"query": queries[qid], "fields": ["title", "text"]}},
                       "size": 30, "_source": ["title", "text"]})
    shorts[qid] = [(h["_id"], h["_source"].get("title", ""), h["_source"].get("text", "")) for h in res["hits"]["hits"]]
    need.update(d[0] for d in shorts[qid])
print(f"shortlists ready: {len(pilot)} queries, {len(need)} distinct docs")

LOG = open("scifact-raw-fixed.jsonl", "w")
rows = []
t00 = time.time()
for n, (qid, rel) in enumerate(pilot):
    docs = {i: (t, x) for i, (idd, t, x) in enumerate(shorts[qid])}
    t0 = time.time()
    r = jev_fixed(queries[qid], docs)
    wall = time.time() - t0
    if "_err" in r or "answers" not in r:
        print(f"  qid {qid} FAILED: {str(r)[:200]}"); rows.append({"qid": qid, "err": str(r)[:200]}); continue
    probs = {int(k[1:]): v.get("noul") for k, v in r["answers"].items() if isinstance(v, dict)}
    order = sorted(probs.keys(), key=lambda i: (-probs[i], i))          # tie -> engine order
    ranked = [shorts[qid][i][0] for i in order] + [shorts[qid][i][0] for i in range(len(shorts[qid])) if i not in probs]
    row = {"qid": qid, "ndcg": ndcg10(ranked, rel), "answered": len(probs), "wall_s": round(wall, 2),
           "in_tok": (r.get("usage") or {}).get("input_tokens"), "out_tok": (r.get("usage") or {}).get("output_tokens"),
           "model": r.get("model"), "probs": {shorts[qid][i][0]: probs[i] for i in probs}}
    rows.append(row); LOG.write(json.dumps({k: v for k, v in row.items() if k != "probs"}) + "\n"); LOG.flush()
    if n % 10 == 0: print(f"  {n+1}/{len(pilot)} ({time.time()-t00:.0f}s)", flush=True)

ok = [r for r in rows if "ndcg" in r]
bm = {json.loads(l)["qid"]: json.loads(l)["bm25"] for l in open("scifact-rerank-pilot.jsonl")}
mb = sum(bm[r["qid"]] for r in ok) / len(ok)
mj = sum(r["ndcg"] for r in ok) / len(ok)
wins = sum(1 for r in ok if r["ndcg"] > bm[r["qid"]]); losses = sum(1 for r in ok if r["ndcg"] < bm[r["qid"]])
ti = sum(r["in_tok"] or 0 for r in ok); to = sum(r["out_tok"] or 0 for r in ok)
print(f"\nqueries: {len(ok)} ok / {len(rows)-len(ok)} failed")
print(f"bm25 (pilot log)   nDCG@10 = {mb:.4f}")
print(f"bm25+jev FIXED     nDCG@10 = {mj:.4f}   (wins {wins} / losses {losses} / ties {len(ok)-wins-losses})")
print(f"usage: input {ti} tok -> ${ti*0.042/1e6:.4f} at $0.042/Mtok input-only")
print(f"models seen: {set(r.get('model') for r in ok)}")

# determinism probe on the fixed shape
print("\ndeterminism probe (3 queries re-asked, fixed shape):")
for qid, rel in pilot[:3]:
    docs = {i: (t, x) for i, (idd, t, x) in enumerate(shorts[qid])}
    r2 = jev_fixed(queries[qid], docs)
    p1 = next(x for x in rows if x["qid"] == qid)["probs"]
    p2 = {shorts[qid][int(k[1:])][0]: v.get("noul") for k, v in r2.get("answers", {}).items() if isinstance(v, dict)}
    common = set(p1) & set(p2)
    dmax = max(abs(p1[k] - p2[k]) for k in common) if common else None
    same = sum(1 for k in common if p1[k] == p2[k])
    print(f"  {qid}: {same}/{len(common)} identical, max |delta| = {dmax}")
