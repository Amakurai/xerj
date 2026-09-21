"""AG News vote eval against one node, keep-alive, with progress.
Usage: vote_eval.py <port> — prints acc / no-hit / conf>=0.8 / ms-item."""
import http.client, json, time, collections, sys

port = int(sys.argv[1])
test = [l.rstrip("\n").split("\t", 1) for l in open("/root/xerj-bench/agnews-test.tsv").readlines()[1:]]
c = http.client.HTTPConnection("localhost", port, timeout=300)
cc = []; none = 0
t0 = time.time()
for n, (gold, txt) in enumerate(test):
    body = json.dumps({"size": 10, "_source": ["label"], "query": {"match": {"text": txt}}}).encode()
    c.request("POST", "/agnews/_search", body, {"content-type": "application/json"})
    r = json.loads(c.getresponse().read())
    hs = r.get("hits", {}).get("hits", [])
    if not hs:
        none += 1; cc.append((0.0, 0)); continue
    v = collections.Counter()
    for i, h in enumerate(hs): v[h["_source"]["label"]] += 1.0 / (i + 1)
    lab, w = v.most_common(1)[0]; conf = w / sum(v.values())
    cc.append((conf, 1 if lab == gold else 0))
    if (n + 1) % 500 == 0:
        el = time.time() - t0
        print(f"  {n+1}/{len(test)}  {el/(n+1)*1000:.2f} ms/item", flush=True)
c.close()
ms = (time.time() - t0) / len(cc) * 1000
acc = sum(k for _, k in cc) / len(cc)
hi = [k for c, k in cc if c >= 0.8]
print(f"port {port}: acc={acc:.4f}  no-hit={none}  conf>=0.8: {len(hi)/len(cc):.1%} at {sum(hi)/max(len(hi),1):.4f}  {ms:.2f} ms/item")
