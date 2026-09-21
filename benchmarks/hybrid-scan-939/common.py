"""Shared helpers for the #939 exact-scan benchmark. Standard library only."""
import hashlib
import json
import os
import urllib.error
import urllib.request

API_KEY = os.environ.get("XERJ_API_KEY", "")


def http(url, method, path, body=None, content_type="application/json", timeout=600, raw=False):
    data = body if isinstance(body, (bytes, type(None))) else json.dumps(body).encode()
    headers = {"content-type": content_type}
    if API_KEY:
        headers["authorization"] = "ApiKey " + API_KEY
    req = urllib.request.Request(url + path, data=data, method=method, headers=headers)
    try:
        payload = urllib.request.urlopen(req, timeout=timeout).read()
    except urllib.error.HTTPError as e:
        return {"_err": e.code, "body": e.read().decode()[:400]}
    return payload.decode() if raw else json.loads(payload)


def loadavg():
    with open("/proc/loadavg") as f:
        return f.read().split()[0]


def cache_counters(url):
    """(hits, misses) of the result cache, scraped from the node's own gauges.

    A run that claims to have measured the engine must show hits did not move.
    """
    text = http(url, "GET", "/v1/metrics", raw=True)
    if isinstance(text, dict):
        raise SystemExit(f"cannot read /v1/metrics: {text}")
    found = {}
    for line in text.splitlines():
        for name in ("xerj_query_cache_hits", "xerj_query_cache_misses"):
            if line.startswith(name + " ") or line.startswith(name + "{"):
                found[name] = found.get(name, 0) + int(float(line.rsplit(" ", 1)[1]))
    if len(found) != 2:
        raise SystemExit(f"result-cache gauges missing from /v1/metrics: {found}")
    return found["xerj_query_cache_hits"], found["xerj_query_cache_misses"]


def distinct_queries(data_dir, n):
    """First n DISTINCT query texts, in file order."""
    seen, out = set(), []
    with open(data_dir + "/queries.jsonl") as f:
        for line in f:
            text = json.loads(line)["text"]
            if text not in seen:
                seen.add(text)
                out.append(text)
                if len(out) == n:
                    break
    if len(out) < n:
        raise SystemExit(f"only {len(out)} distinct queries in {data_dir}, need {n}")
    return out


def bucket_of(doc_id):
    """Deterministic 10-way keyword partition of the corpus, for the filter arms."""
    return "b%d" % (int(hashlib.md5(doc_id.encode()).hexdigest(), 16) % 10)


def n_of(doc_id):
    """Deterministic integer in [0, 1000), for the range-filter arm."""
    return int(hashlib.md5(("n" + doc_id).encode()).hexdigest(), 16) % 1000


def bm25_query(q):
    return {"multi_match": {"query": q, "fields": ["title", "text"]}}


def semantic_query(q, k=10, flt=None):
    body = {"field": "body", "query": q, "k": k}
    if flt is not None:
        body["filter"] = flt
    return {"semantic": body}


def hybrid_query(q, k=10):
    return {"hybrid": {"queries": [{"query": bm25_query(q)}, {"query": semantic_query(q, k)}], "fusion": "rrf"}}


def pct(sorted_values, p):
    return sorted_values[min(len(sorted_values) - 1, int(len(sorted_values) * p))]
