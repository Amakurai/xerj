#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# ingest-docs-per-second.sh — one-command reproduction of the ingest headline.
#
#   CLAIMED   XERJ 191,286 docs/s on the "ingest 100k x c1" cell
#             (ES 8.13.4 on the identical corpus and box: 111,073 docs/s,
#             1.72x) — demo/playbooks/SCORECARD.md:17, quoted on
#             landing/benchmarks/elasticsearch.html ("Bulk ingest, identical
#             corpus and box: 1.72x higher").
#
# What this script does: builds nothing, argues nothing — it starts a
# throwaway node on a PRIVATE port (93xx, never :9200), indexes 100,000
# synthetic LLM-telemetry docs through ES-wire _bulk exactly the way the
# scorecard harness does (1 client, 10k docs/batch, pre-generated batch files
# so file-write cost stays out of the timed window, docs/s computed from the
# VERIFIED post-refresh count, not the intended count), and prints the number
# next to the claim.
#
# The ES side of the comparison is NOT run here (it needs an ES install);
# demo/playbooks/bench-matrix.mjs --ingest-only is the head-to-head harness.
#
# Usage:
#   bash scripts/bench/ingest-docs-per-second.sh
# Knobs: DOCS=100000 BATCH=10000 PORT=<free 93xx port> XERJ_BIN=<binary> KEEP=1
# Requires: curl, python3, a built server binary (engine/target/release/xerj).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/repro.sh
source "$HERE/lib/repro.sh"

DOCS="${DOCS:-100000}"
BATCH="${BATCH:-10000}"
CORPUS="$REPO_ROOT/demo/data/extras/chat-events.ndjson"

repro_find_binary
repro_free_port 9310
PORT_NOTE="$REPRO_PORT"

repro_header "one-command repro: bulk ingest docs/s"
repro_claim  "191,286 docs/s — ingest 100k x c1, scorecard run 2026-07 (demo/playbooks/SCORECARD.md:17)"
repro_claim  "            (ES 8.13.4 same corpus/box: 111,073 docs/s -> 1.72x; ES side needs an ES install, not run here)"

# ── corpus: the committed synthetic telemetry the published run cycled ──────
if [ ! -f "$CORPUS" ]; then
  echo "corpus $CORPUS missing — generating an equivalent-shaped fallback"
  CORPUS="$(mktemp /tmp/xerj-bench-corpus.XXXXXX.ndjson)"
  python3 - "$CORPUS" <<'PY'
import json, random, sys, time
random.seed(42)
out = open(sys.argv[1], "w")
models = ["claude-sonnet-4-5"] * 5 + ["claude-haiku-4-5"] * 3 + ["gpt-5.2", "gemini-3-pro"]
intents = ["refactor", "explain", "debug", "review", "test", "migrate", "optimise",
           "document", "design", "triage", "revert", "ship"]
tops = ["AGENTS.md", "engine/CLAUDE.md", "docs/RERANK.md", "xerj.default.toml",
        "landing/index.html", "scripts/bench", "crates/xerj-engine/src/index.rs",
        "Cargo.toml", "README.md", "es-compat", "wal", "hnsw"]
tenants = ["acme", "globex", "initech", "umbrella", "wayne"]
base = time.time()
for i in range(4008):
    pt = max(1, int(random.gauss(900, 300)))
    ct = max(0, int(random.gauss(250, 90)))
    comp = max(0, int(random.gauss(180, 70)))
    lat = max(1, int(random.gauss(random.choice([310, 520, 780, 1100, 1450]), 120)))
    out.write(json.dumps({
        "@timestamp": base + i, "model": random.choice(models), "intent": random.choice(intents),
        "top_doc": random.choice(tops), "tenant": random.choice(tenants),
        "status": "error" if random.random() < 0.0125 else "ok", "cache_hit": random.random() < 0.429,
        "prompt_tokens": pt, "context_tokens": ct, "completion_tokens": comp,
        "latency_ms": lat, "cost_usd": round((pt + ct + comp) * 3e-6, 6),
    }) + "\n")
PY
  repro_note "fallback corpus generated (the committed one is what the published run used)"
fi

RAW_LINES="$(wc -l <"$CORPUS")"
repro_note "corpus: $CORPUS ($RAW_LINES distinct events, cycled to $DOCS docs — the published protocol)"

repro_node_start -
trap repro_node_stop EXIT
repro_node_wait

BASE="http://127.0.0.1:$REPRO_PORT"
IDX="bench-ingest"

curl -sf -XDELETE "$BASE/$IDX" >/dev/null 2>&1 || true
curl -sf -XPUT "$BASE/$IDX" -H 'content-type: application/json' --data-binary @- >/dev/null <<'JSON'
{ "mappings": { "properties": {
  "@timestamp": { "type": "date" },
  "model": { "type": "keyword" }, "intent": { "type": "keyword" }, "status": { "type": "keyword" },
  "tenant": { "type": "keyword" }, "top_doc": { "type": "keyword" }, "cache_hit": { "type": "boolean" },
  "prompt_tokens": { "type": "integer" }, "context_tokens": { "type": "integer" }, "completion_tokens": { "type": "integer" },
  "latency_ms": { "type": "integer" }, "cost_usd": { "type": "double" }
} } }
JSON

# ── pre-generate every batch file, then time only the _bulk posts ───────────
SCRATCH="$(mktemp -d /tmp/xerj-bench-bulk.XXXXXX)"
trap 'rm -rf "$SCRATCH"; repro_node_stop' EXIT
python3 - "$CORPUS" "$SCRATCH" "$DOCS" "$BATCH" <<'PY'
import os, sys
corpus, scratch, total, batch = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
raw = open(corpus).read().splitlines()
n, written, b = len(raw), 0, 0
while written < total:
    lines = []
    for _ in range(min(batch, total - written)):
        lines.append('{"index":{}}')
        lines.append(raw[written % n])
        written += 1
    with open(os.path.join(scratch, f"batch_{b:04d}.ndjson"), "w") as f:
        f.write("\n".join(lines) + "\n")
    b += 1
print(f"batches: {b} x {batch} docs (pre-generated, out of the timed window)")
PY

bulk_errors=0
bulk_item_errors=0
start_ns=$(date +%s%N)
for f in "$SCRATCH"/batch_*.ndjson; do
  resp="$(curl -s -XPOST "$BASE/$IDX/_bulk" -H 'content-type: application/x-ndjson' --data-binary @"$f")"
  # a silent drop must not be able to inflate docs/s: read every response
  if ! errcount="$(printf '%s' "$resp" | python3 -c '
import json, sys
try:
    j = json.load(sys.stdin)
except Exception:
    print("PARSE"); sys.exit()
if j.get("errors"):
    bad = sum(1 for it in j.get("items", []) if (it and next(iter(it.values()), {}) or {}).get("status", 0) >= 400)
    print(bad or "ITEMS")
else:
    print(0)
')"; then
    bulk_errors=$((bulk_errors + 1))
    continue
  fi
  case "$errcount" in
    0) ;;
    PARSE|ITEMS) bulk_errors=$((bulk_errors + 1)) ;;
    *) bulk_errors=$((bulk_errors + 1)); bulk_item_errors=$((bulk_item_errors + errcount)) ;;
  esac
done
end_ns=$(date +%s%N)

curl -sf -XPOST "$BASE/$IDX/_refresh" >/dev/null
count="$(curl -sf "$BASE/$IDX/_count" | python3 -c 'import json,sys; print(json.load(sys.stdin)["count"])')"
elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
dps="$(python3 -c "print(round($count / ($elapsed_ms / 1000)))")"

printf '\n'
repro_measured "$dps docs/s ($count docs indexed and verified by _count in ${elapsed_ms} ms, 1 client, $BATCH-doc batches)"
repro_measured "bulk responses with errors: $bulk_errors (item errors: $bulk_item_errors)"
python3 - "$dps" <<'PY'
import sys
dps, claimed = int(sys.argv[1]), 191286
r = dps / claimed
print(f"  ratio     {r:.2f}x of the claimed figure (same protocol, this machine — not the scorecard's box)")
print(f"            vs the ES side of the published cell (111,073 docs/s): {dps / 111073:.2f}x")
PY
repro_note "docs/s uses the verified post-refresh count, never the intended count (scorecard rule)"
repro_env_box
repro_note "teardown: node stopped, throwaway data dir removed (KEEP=1 to keep)"
