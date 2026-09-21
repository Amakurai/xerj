#!/usr/bin/env python3
"""Local test double for the Jev rerank provider (TypeSafe AI System One).

XERJ's `rerank` search stage POSTs one request per judged document batch to
the provider's `/v1/systemone`. This stub speaks that wire format so the
stage's published cost/concurrency facts can be measured end-to-end against a
real node with no key, no network egress, and no non-determinism:

  request   {"model": ..., "state": {"query": ..., "documents":
             {"d0": {"title": ..., "text": ...}, ...}}}
  response  {"model": ..., "answers": {"d0": {"type": "noul", "noul": 0.42}},
             "usage": {"input_tokens": N, "output_tokens": N}}

Verdicts are the same deterministic word-overlap relevance the engine's own
test double uses (engine/crates/xerj-api/tests/rerank_stage_http.rs
`Verdicts::Overlap`): p = |query words found in title+text| / |query words|.

Control (a JSON file watched per request, so a running repro can flip modes):
  {"sleep_ms": 2000}   stall every call, to exercise the degrade-on-deadline
                       path and the in-flight ceiling

Every call is appended to a JSONL log: one line with start/end monotonic
timestamps, the document count, the batch keys, and the question. Call counts,
per-call document counts and max in-flight concurrency are computed from that
log by the caller. The Authorization header is NEVER logged.

Usage: jev_stub.py --port 9340 --log calls.jsonl --control control.json
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def words(s):
    return {w for w in "".join(c if c.isalnum() else " " for c in s.lower()).split() if w}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the repro's stdout clean
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/v1/systemone":
            return self._json(404, {"error": "not found"})
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw)
        except Exception:
            return self._json(400, {"error": "bad json"})
        start = time.monotonic()

        try:
            with open(self.server.control_path) as f:
                sleep_ms = float(json.load(f).get("sleep_ms", 0))
        except Exception:
            sleep_ms = 0.0
        if sleep_ms:
            time.sleep(sleep_ms / 1000.0)

        state = parsed.get("state") or {}
        qwords = words(str(state.get("query") or ""))
        docs = state.get("documents") or {}
        answers = {}
        for key, doc in docs.items():
            have = words(f"{doc.get('title', '')} {doc.get('text', '')}")
            p = (len(qwords & have) / len(qwords)) if qwords else 0.0
            answers[key] = {"type": "noul", "noul": round(p, 6)}
        self._json(200, {
            "model": parsed.get("model", "jev-latest"),
            "answers": answers,
            "usage": {"input_tokens": 100, "output_tokens": 10},
        })
        end = time.monotonic()
        with open(self.server.log_path, "a") as f:
            f.write(json.dumps({
                "seq": int(time.time() * 1000) % 10**9,
                "start": start,
                "end": end,
                "ndocs": len(docs),
                "keys": sorted(docs.keys()),
                "query": str(state.get("query") or "")[:200],
            }) + "\n")

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {"ok": True})
        return self._json(404, {"error": "not found"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--control", required=True)
    args = ap.parse_args()
    open(args.log, "w").close()
    json.dump({"sleep_ms": 0}, open(args.control, "w"))
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    srv.log_path = args.log
    srv.control_path = args.control
    srv.serve_forever()


if __name__ == "__main__":
    main()
