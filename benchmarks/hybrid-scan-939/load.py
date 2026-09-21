#!/usr/bin/env python3
"""Load a BEIR corpus into a throwaway XERJ node.

    XERJ_API_KEY=... python3 load.py http://localhost:10840 scifact /path/to/beir/scifact [clients]

WRITES: it deletes and recreates the index named on the command line. Point it
only at a node you started yourself, on a port you own.

`body` is `semantic_text` (title + ". " + text, as the BEIR harness this was
copied from built it), so every document longer than 512 characters is stored
as several passage vectors — the shape issue #939 is about. `bucket` and `n`
are deterministic functions of `_id` that give the filter arms something to
select on. `clients` > 1 sends `_bulk` requests concurrently (useful under
`--embed-mode neural`, see #938); document order inside the index then depends
on arrival order, so build the index ONCE and run both binaries over that same
data directory when comparing results.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from common import bucket_of, http, loadavg, n_of

url, index, data_dir = sys.argv[1], sys.argv[2], sys.argv[3]
clients = int(sys.argv[4]) if len(sys.argv) > 4 else 1
BULK = 100

print("delete:", http(url, "DELETE", "/" + index))
print("create:", http(url, "PUT", "/" + index, {"mappings": {"properties": {
    "title": {"type": "text"}, "text": {"type": "text"},
    "bucket": {"type": "keyword"}, "n": {"type": "integer"},
    "body": {"type": "semantic_text"}}}}))

docs = [json.loads(line) for line in open(data_dir + "/corpus.jsonl")]


def send(start):
    lines = []
    for d in docs[start:start + BULK]:
        lines.append(json.dumps({"index": {"_index": index, "_id": d["_id"]}}))
        lines.append(json.dumps({
            "title": d["title"], "text": d["text"],
            "bucket": bucket_of(d["_id"]), "n": n_of(d["_id"]),
            "body": d["title"] + ". " + d["text"]}))
    r = http(url, "POST", "/_bulk", ("\n".join(lines) + "\n").encode(), "application/x-ndjson", timeout=3600)
    if r.get("errors") or "_err" in r:
        raise SystemExit("bulk problem: " + str(r)[:400])
    return len(lines) // 2


t0 = time.time()
done = 0
with ThreadPoolExecutor(max_workers=clients) as pool:
    for n in pool.map(send, range(0, len(docs), BULK)):
        done += n
        if done % 1000 < BULK:
            print(f"indexed {done}/{len(docs)}  {done / (time.time() - t0):.1f} docs/s  loadavg={loadavg()}", flush=True)
print("refresh:", http(url, "POST", f"/{index}/_refresh"))
print("flush:", http(url, "POST", f"/{index}/_flush"))
print("count:", http(url, "GET", f"/{index}/_count"))
print(f"wall={time.time() - t0:.1f}s clients={clients} loadavg={loadavg()}")
