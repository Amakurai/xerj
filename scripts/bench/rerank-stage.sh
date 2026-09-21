#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# rerank-stage.sh — one-command reproduction of the Jev rerank-stage numbers.
#
# The stage's RANKING QUALITY is not claimable — XERJ has never run the real
# Jev model (no provider key; docs/RERANK.md says so on the tin, and the BEIR
# figures for Jev/Voyage/Cohere are that other project's, marked "not run by
# us"). What public copy DOES claim about this stage is its cost and failure
# contract, and those are measurable against a local test double with zero
# network egress:
#
#   CLAIMED   "30 documents per provider call ... window: 35 is two calls
#              (30 + 5), and window: 300, the maximum, is ten"
#              "8 calls in flight by default, 16 at most"
#              "There is no verdict cache: three page requests (from 0, 10, 20)
#              over one 30-document window are three provider calls and 90 paid
#              judgements, not 30"
#              "a slow provider -> HTTP 200 with the engine's order and
#              _rerank.applied:false"
#              "no key -> 503 ... all rerank_exception with no hits"
#              — docs/RERANK.md (Cost and concurrency facts / Failure policy),
#              restated on landing/benchmarks/index.html and landing/llms.txt.
#
# What this script does: starts a Jev test double (lib/jev_stub.py — the same
# word-overlap verdicts as the engine's own double) on a PRIVATE port, boots a
# node pointed at it with a dummy key, indexes 320 generated docs, and measures
# each published fact end-to-end through the ES wire. Every provider call the
# node makes lands in a log this script reads back: call counts, per-call
# document counts, max in-flight concurrency, and judgements billed.
#
# Usage:
#   bash scripts/bench/rerank-stage.sh
# Knobs: PORT=<free 93xx port> STUB_PORT=<free 93xx port> XERJ_BIN=... KEEP=1
# Requires: python3, curl, a built server binary. No network egress: the
# provider endpoint is 127.0.0.1.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/repro.sh
source "$HERE/lib/repro.sh"

repro_find_binary
repro_free_port 9340
NODE_PORT="$REPRO_PORT"
repro_free_port 9350
STUB_PORT="$REPRO_PORT"

repro_header "one-command repro: Jev rerank-stage cost/failure contract"
repro_claim  "30 docs per provider call — window 35 -> 2 calls (30+5), window 300 -> 10 (docs/RERANK.md)"
repro_claim  "no verdict cache — three page requests over one window = 3 calls, 90 paid judgements"
repro_claim  "8 calls in flight by default, 16 at most; slow provider -> HTTP 200, applied:false, engine order"
repro_claim  "no key -> HTTP 503, rerank_exception, no hits; ranking quality with the real model: NOT claimed"
repro_note   "verdicts come from lib/jev_stub.py (word overlap), not the real model — the quality rows stay unreproducible by design"

WORK="$(mktemp -d /tmp/xerj-bench-rerank.XXXXXX)"
STUB_LOG="$WORK/calls.jsonl"
STUB_CONTROL="$WORK/control.json"
CONFIG="$WORK/rerank.toml"
KEYLESS_PID=""
STUB_PID=""
cleanup() {
  for pid in "$STUB_PID" "$KEYLESS_PID"; do
    [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
  done
  for pid in "$STUB_PID" "$KEYLESS_PID"; do
    [ -n "$pid" ] || continue
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
  done
  [ -n "${KEYLESS_DATA:-}" ] && rm -rf "$KEYLESS_DATA"
  repro_node_stop
  rm -rf "$WORK"
}
trap cleanup EXIT

# ── the provider test double ────────────────────────────────────────────────
python3 "$HERE/lib/jev_stub.py" --port "$STUB_PORT" --log "$STUB_LOG" --control "$STUB_CONTROL" &
STUB_PID=$!
for _ in $(seq 1 50); do
  curl -sf "http://127.0.0.1:$STUB_PORT/health" >/dev/null 2>&1 && break
  sleep 0.1
done

# ── the node, pointed at the double ─────────────────────────────────────────
# repro_node_start reads the REPRO_PORT global — restore the node's port after
# the stub's allocation changed it.
REPRO_PORT="$NODE_PORT"
cat >"$CONFIG" <<TOML
[rerank]
enabled  = true
api_key  = "stub-key-for-bench-only"
endpoint = "http://127.0.0.1:$STUB_PORT/v1/systemone"
TOML
repro_node_start "$CONFIG"
repro_node_wait
BASE="http://127.0.0.1:$NODE_PORT"
IDX="bench-rerank"

curl -sf -XDELETE "$BASE/$IDX" >/dev/null 2>&1 || true
curl -sf -XPUT "$BASE/$IDX" -H 'content-type: application/json' \
  --data-binary '{"mappings":{"properties":{"title":{"type":"text"},"body":{"type":"text"}}}}' >/dev/null

# ── measurement driver (writes $WORK/results.json, prints nothing) ──────────
python3 - "$BASE" "$IDX" "$STUB_LOG" "$STUB_CONTROL" "$WORK/results.json" <<'PY'
import http.client, json, subprocess, sys

base, idx, log_path, control_path, out_path = sys.argv[1:6]
host, port = "127.0.0.1", int(base.rsplit(":", 1)[1])
QUESTION = "vitamin d supplementation bone density"
WORDS = QUESTION.split()

def call(method, path, body=None):
    c = http.client.HTTPConnection(host, port, timeout=40)
    payload = json.dumps(body) if body is not None else None
    headers = {"content-type": "application/json"} if payload else {}
    c.request(method, path, body=payload, headers=headers)
    r = c.getresponse()
    raw = r.read()
    c.close()
    try:
        return r.status, json.loads(raw)
    except Exception:
        return r.status, raw.decode(errors="replace")[:400]

def set_sleep(ms):
    json.dump({"sleep_ms": ms}, open(control_path, "w"))

def log_lines():
    try:
        return [json.loads(l) for l in open(log_path) if l.strip()]
    except FileNotFoundError:
        return []

def calls_since(marker):
    return log_lines()[marker:]

def mark():
    return len(log_lines())

def max_in_flight(calls):
    events = []
    for c in calls:
        events += [(c["start"], 1), (c["end"], -1)]
    events.sort()
    cur = best = 0
    for _, d in events:
        cur += d
        best = max(best, cur)
    return best

# corpus: 320 docs whose word overlap with the question is a deterministic ramp
# (320 > the 300-doc max window so the window arithmetic is measured at its
# published maximum, not truncated by the corpus size)
nd = ""
for i in range(320):
    keep = (320 - i) % 6                      # 0..5 of the 5 question words
    body = " ".join(WORDS[:keep]) + f" study report {i} filler text about trials"
    nd += '{"index":{}}\n' + json.dumps({"title": f"Trial report {i}", "body": body}) + "\n"
open(f"{out_path}.ndjson", "w").write(nd)
subprocess.run(["curl", "-s", "-XPOST", f"{base}/{idx}/_bulk",
                "-H", "content-type: application/x-ndjson",
                "--data-binary", f"@{out_path}.ndjson"], check=True,
               stdout=subprocess.DEVNULL)
call("POST", f"/{idx}/_refresh")
status, cnt = call("GET", f"/{idx}/_count")
indexed = (cnt or {}).get("count")
if indexed != 320:
    # a silent ingest failure would make every later check vacuously pass on
    # an empty index — refuse to measure instead
    json.dump([("corpus ingest", "320 docs", f"{indexed} docs indexed", False)], open(out_path, "w"))
    raise SystemExit(0)

def search(rerank, size=10, frm=0):
    # first stage is match_all on purpose: the question lives in rerank.query,
    # and a match on a term the analyzer does not return hits for would make
    # every check below vacuously pass on an empty hit list.
    body = {"query": {"match_all": {}}, "size": size, "from": frm}
    if rerank is not None:
        body["rerank"] = rerank
    return call("POST", f"/{idx}/_search", body)

results = []   # rows: (claim label, claimed value, measured value, ok|None)

# A — configuration facts as the live node reports them
status, cfg = call("GET", "/_xerj/rerank")
if status == 200:
    d, lim = cfg.get("defaults", {}), cfg.get("limits", {})
    results.append(("defaults window/batch/max_concurrency", "30 / 30 / 8",
                    f"{d.get('window')} / {d.get('batch')} / {d.get('max_concurrency')}",
                    (d.get("window"), d.get("batch"), d.get("max_concurrency")) == (30, 30, 8)))
    results.append(("limits max_docs_per_call/max_window/max_concurrency", "30 / 300 / 16",
                    f"{lim.get('max_docs_per_call')} / {lim.get('max_window')} / {lim.get('max_concurrency')}",
                    (lim.get("max_docs_per_call"), lim.get("max_window"), lim.get("max_concurrency")) == (30, 300, 16)))
else:
    results.append(("GET /_xerj/rerank", "200", f"HTTP {status}", False))

# B — window 35 -> 2 calls (30 + 5)
m = mark(); search({"window": 35, "query": QUESTION}, size=5)
c = calls_since(m)
results.append(("window 35 -> provider calls / docs per call", "2 calls: 30 + 5",
                f"{len(c)} calls: {' + '.join(str(x['ndocs']) for x in c)}",
                sorted((x["ndocs"] for x in c), reverse=True) == [30, 5]))

# C — window 300 (the maximum) -> 10 calls x 30
m = mark(); search({"window": 300, "query": QUESTION}, size=5)
c = calls_since(m)
results.append(("window 300 -> provider calls / docs per call", "10 calls of 30",
                f"{len(c)} calls: {' + '.join(str(x['ndocs']) for x in c)}",
                len(c) == 10 and all(x["ndocs"] == 30 for x in c)))

# D — no verdict cache: three page requests over one 30-doc window
m = mark()
for frm in (0, 10, 20):
    search({"window": 30, "query": QUESTION}, size=10, frm=frm)
c = calls_since(m)
judged = sum(x["ndocs"] for x in c)
results.append(("3 page requests (from 0/10/20) -> calls / paid judgements",
                "3 calls, 90 judgements — not 30",
                f"{len(c)} calls, {judged} judgements",
                len(c) == 3 and judged == 90))

# E — slow provider: degrade on deadline, HTTP 200, the engine's order back
_, plain = search(None, size=10)
engine_order = [h["_id"] for h in plain["hits"]["hits"]]
if len(engine_order) < 10:
    json.dump([("first-stage query", ">= 10 hits to judge",
                f"{len(engine_order)} hits — corpus or query wrong", False)], open(out_path, "w"))
    raise SystemExit(0)
set_sleep(2000)
status, deg = search({"window": 30, "query": QUESTION, "timeout_ms": 200}, size=10)
set_sleep(0)
rr = deg.get("_rerank", {}) if isinstance(deg, dict) else {}
deg_order = [h["_id"] for h in deg.get("hits", {}).get("hits", [])] if isinstance(deg, dict) else []
ok = (status == 200 and rr.get("applied") is False and rr.get("score_kind") == "engine"
      and deg_order == engine_order)
results.append(("slow provider (2s) with timeout_ms=200",
                "HTTP 200, applied:false, score_kind:engine, engine order, same totals",
                f"HTTP {status}, applied:{rr.get('applied')}, score_kind:{rr.get('score_kind')}, "
                f"order {'== engine' if deg_order == engine_order else 'DIFFERS'}, reason: {str(rr.get('reason'))[:48]}",
                ok))

# F — verdicts replace _score, order by probability, min_score prunes
status, applied = search({"window": 40, "query": QUESTION, "min_score": 0.5}, size=10)
rr = applied.get("_rerank", {})
hits = applied["hits"]["hits"]
scores = [h["_score"] for h in hits]
sorted_desc = scores == sorted(scores, reverse=True)
ok = (status == 200 and rr.get("score_kind") == "probability" and sorted_desc
      and scores and min(scores) >= 0.5 and (rr.get("pruned_below_min_score") or 0) > 0)
results.append(("verdicts replace _score (window 40, min_score 0.5)",
                "score_kind:probability, hits ordered by probability, all >= min_score, prunes counted",
                f"score_kind:{rr.get('score_kind')}, judged:{rr.get('judged')}, "
                f"pruned_below_min_score:{rr.get('pruned_below_min_score')}, "
                f"order {'desc' if sorted_desc else 'NOT SORTED'}, min _score {min(scores):.3f}" if scores else "no hits",
                ok))

# G — in-flight ceiling: 8 by default, a request can raise it to at most 16
set_sleep(400)
m = mark(); search({"window": 300, "query": QUESTION, "timeout_ms": 30000}, size=5)
c8 = calls_since(m)
m = mark(); search({"window": 300, "query": QUESTION, "timeout_ms": 30000, "max_concurrency": 16}, size=5)
c16 = calls_since(m)
set_sleep(0)
f8, f16 = max_in_flight(c8), max_in_flight(c16)
results.append(("max provider calls in flight, window 300 (10 calls)",
                "8 by default, 16 at most",
                f"default: {f8} in flight over {len(c8)} calls; max_concurrency=16: {f16} in flight over {len(c16)} calls",
                1 <= f8 <= 8 and 8 < f16 <= 16))

json.dump(results, open(out_path, "w"))
PY

# ── the keyless node: no key configured -> 503, no hits ─────────────────────
KEYLESS_DATA="$(mktemp -d /tmp/xerj-bench-keyless.XXXXXX)"
repro_free_port 9360
KEYLESS_PORT="$REPRO_PORT"
REPRO_PORT="$NODE_PORT"   # keep the env box honest: this repro's main node
env "$XERJ_BIN" --insecure --port "$KEYLESS_PORT" --data-dir "$KEYLESS_DATA" >"$KEYLESS_DATA/log" 2>&1 &
KEYLESS_PID=$!
for _ in $(seq 1 100); do
  curl -sf "http://127.0.0.1:$KEYLESS_PORT/" >/dev/null 2>&1 && break
  sleep 0.2
done
# the keyless node needs the index too — a 404 would test routing, not rerank
curl -sf -XPUT "http://127.0.0.1:$KEYLESS_PORT/$IDX" -H 'content-type: application/json' \
  --data-binary '{"mappings":{"properties":{"title":{"type":"text"},"body":{"type":"text"}}}}' >/dev/null
printf '{"index":{}}\n{"title":"Trial report 0","body":"trial filler text"}\n' \
  | curl -sf -XPOST "http://127.0.0.1:$KEYLESS_PORT/$IDX/_bulk" \
    -H 'content-type: application/x-ndjson' --data-binary @- >/dev/null
curl -sf -XPOST "http://127.0.0.1:$KEYLESS_PORT/$IDX/_refresh" >/dev/null

KEYLESS_OUT="$(python3 - "http://127.0.0.1:$KEYLESS_PORT" "$IDX" <<'PY'
import http.client, json, sys
base, idx = sys.argv[1], sys.argv[2]
host, port = "127.0.0.1", int(base.rsplit(":", 1)[1])
c = http.client.HTTPConnection(host, port, timeout=15)
body = {"query": {"match": {"body": "trial"}}, "size": 5,
        "rerank": {"query": "vitamin d supplementation bone density"}}
c.request("POST", f"/{idx}/_search", body=json.dumps(body), headers={"content-type": "application/json"})
r = c.getresponse()
raw = r.read().decode(errors="replace")
c.close()
try:
    j = json.loads(raw)
    hits = len((j.get("hits") or {}).get("hits") or [])
    err = j.get("error")
    if isinstance(err, dict):
        etype = err.get("type") or (err.get("caused_by") or {}).get("type") or ""
    else:
        etype = str(err)[:80]
    print(json.dumps({"status": r.status, "hits": hits, "error_type": etype, "raw_type": raw[:0]}))
except Exception:
    print(json.dumps({"status": r.status, "raw": raw[:200]}))
PY
)"

# ── verdicts ────────────────────────────────────────────────────────────────
python3 - "$WORK/results.json" "$KEYLESS_OUT" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
keyless = json.loads(sys.argv[2])
print()
fails = 0
for label, claimed, measured, ok in rows:
    verdict = "PASS" if ok else ("FAIL" if ok is False else "----")
    fails += 1 if ok is False else 0
    print(f"  {verdict}  {label}")
    print(f"         claimed  : {claimed}")
    print(f"         measured : {measured}")
kl_ok = keyless.get("status") == 503 and keyless.get("hits") == 0 and "rerank_exception" in str(keyless.get("error_type", ""))
fails += 0 if kl_ok else 1
print(f"  {'PASS' if kl_ok else 'FAIL'}  no key configured -> refusal, not silence")
print(f"         claimed  : HTTP 503, rerank_exception, no hits")
print(f"         measured : HTTP {keyless.get('status')}, hits {keyless.get('hits')}, error type {keyless.get('error_type') or keyless.get('raw')}")
print()
print(f"  contract checks: {len(rows) + 1 - fails}/{len(rows) + 1} PASS "
      "(deterministic contract — these assertions are the reproduction)")
print('  note      the quality rows (nDCG 0.768/0.358 for Jev, etc.) are hev/jev-rerank\'s numbers,')
print('            marked "not run by us" in docs/RERANK.md — no harness here can or should reproduce them')
PY

repro_env_box
repro_note "teardown: both nodes + the stub stopped, throwaway dirs removed (KEEP=1 to keep)"
