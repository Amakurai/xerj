"""AG News (HF fancyzhx/ag_news): classify the 7,600 test items by a
rank-weighted k=10 BM25 vote over the 120,000 labelled train items.
Same methodology as benchmarks/decisions-as-retrieval/eval.py."""
import json, time, urllib.request, collections
U = "http://localhost:9460"; K = 10
def req(m, p, b=None, ct="application/json"):
    d = b if isinstance(b, (bytes, type(None))) else json.dumps(b).encode()
    r = urllib.request.Request(U + p, data=d, method=m, headers={"content-type": ct})
    try: return json.loads(urllib.request.urlopen(r, timeout=300).read())
    except urllib.error.HTTPError as e: return {"_err": e.code}

req("DELETE", "/agnews")
req("PUT", "/agnews", {"mappings": {"properties": {"text": {"type": "text"}, "label": {"type": "keyword"}}}})
lines = []
for i, l in enumerate(open("agnews-train.tsv").readlines()[1:]):
    lab, txt = l.rstrip("\n").split("\t", 1)
    lines.append(json.dumps({"index": {"_index": "agnews", "_id": str(i)}}))
    lines.append(json.dumps({"text": txt, "label": lab}))
    if len(lines) >= 4000:
        req("POST", "/_bulk", ("\n".join(lines) + "\n").encode(), "application/x-ndjson"); lines = []
    if i % 20000 == 0: print("indexed", i, flush=True)
if lines: req("POST", "/_bulk", ("\n".join(lines) + "\n").encode(), "application/x-ndjson")
print(req("POST", "/agnews/_refresh")); print("count:", req("GET", "/agnews/_count")["count"])

test = [l.rstrip("\n").split("\t", 1) for l in open("agnews-test.tsv").readlines()[1:]]
print("test items:", len(test))
cc = []; t0 = time.time(); none = 0
for gold, txt in test:
    r = req("POST", "/agnews/_search", {"size": K, "_source": ["label"], "query": {"match": {"text": txt}}})
    hs = r.get("hits", {}).get("hits", [])
    if not hs: none += 1; cc.append((0.0, 0)); continue
    v = collections.Counter()
    for i, h in enumerate(hs): v[h["_source"]["label"]] += 1.0 / (i + 1)
    lab, w = v.most_common(1)[0]; conf = w / sum(v.values())
    cc.append((conf, 1 if lab == gold else 0))
acc = sum(k for _, k in cc) / len(cc); ms = (time.time() - t0) / len(cc) * 1000
def ece(cc, bins=10):
    n = len(cc); tot = 0.0
    for b in range(bins):
        xs = [(c, k) for c, k in cc if b / bins < c <= (b + 1) / bins or (b == 0 and c == 0)]
        if xs: tot += len(xs) / n * abs(sum(k for _, k in xs) / len(xs) - sum(c for c, _ in xs) / len(xs))
    return tot
hi = [k for c, k in cc if c >= 0.8]
print(f"AG News 4-way — BM25 vote, k={K}")
print(f"  acc={acc:.4f}  ECE={ece(cc):.3f}  conf>=0.8: {len(hi)/len(cc):.1%} of items at {sum(hi)/max(len(hi),1):.4f} acc  no-hit={none}  {ms:.1f} ms/item")
