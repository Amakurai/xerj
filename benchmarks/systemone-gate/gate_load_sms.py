"""Load the SMS-spam history for the /v1/systemone acceptance gate.

Indexes the same 4,000-message train split benchmarks/decisions-as-retrieval
used (shuffle seed 7, first 4,000) and writes the held-out gate documents to
gate-docs.json: 5 spam + 5 ham, picked for length and signal, so the gate run
has fixed inputs. Usage: python3 gate_load_sms.py [es_compat_url]
"""
import json, random, sys, urllib.request

U = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9440"

def req(m, p, b=None, ct="application/json"):
    d = b if isinstance(b, (bytes, type(None))) else json.dumps(b).encode()
    r = urllib.request.Request(U + p, data=d, method=m, headers={"content-type": ct})
    try:
        return json.loads(urllib.request.urlopen(r, timeout=900).read())
    except urllib.error.HTTPError as e:
        return {"_err": e.code, "body": e.read().decode()[:300]}

rows = [l.rstrip("\n").split("\t", 1) for l in open("sms.tsv")]
rows = [(label, text) for label, text in rows]  # (label, text)
random.Random(7).shuffle(rows)
train, held = rows[:4000], rows[4000:]

print(req("DELETE", "/sms"))
print(req("PUT", "/sms", {"mappings": {"properties": {
    "label": {"type": "keyword"}, "text": {"type": "text"}}}}))
for i in range(0, len(train), 250):
    lines = []
    for j, (label, text) in enumerate(train[i:i + 250]):
        lines.append(json.dumps({"index": {"_index": "sms", "_id": str(i + j)}}))
        lines.append(json.dumps({"text": text, "label": label}))
    r = req("POST", "/_bulk", ("\n".join(lines) + "\n").encode(), "application/x-ndjson")
    if r.get("errors") or "_err" in r:
        print("bulk problem", str(r)[:300]); sys.exit(1)
req("POST", "/sms/_refresh")
print("sms count:", req("GET", "/sms/_count").get("count"))

# Fixed gate documents: distinctive spam and ham, neither near-duplicates of a
# train row's phrasing, each 40..140 chars.
SPAM_KEYS = ["claim", "free", "winner", "prize", "urgent", "cash"]
def pick(label, n):
    out = []
    for text in [t for l, t in held if l == label]:
        if not (40 <= len(text) <= 140):
            continue
        keys = sum(k in text.lower() for k in SPAM_KEYS)
        want = 2 if label == "spam" else 0
        if keys >= want and text not in out:
            out.append(text)
        if len(out) == n:
            break
    return out

docs = [{"label": "spam", "text": t} for t in pick("spam", 5)]
docs += [{"label": "ham", "text": t} for t in pick("ham", 5)]
assert len(docs) == 10, f"picked {len(docs)}"
json.dump(docs, open("gate-docs.json", "w"), indent=1)
print("gate docs written:", sum(d["label"] == "spam" for d in docs), "spam /",
      sum(d["label"] == "ham" for d in docs), "ham")
for d in docs[:2] + docs[5:7]:
    print(" ", d["label"], "|", d["text"][:90])
