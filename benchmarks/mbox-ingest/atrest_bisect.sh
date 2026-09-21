#!/usr/bin/env bash
# Issue #948 experiment: the at-rest retention bisection.
#
# Loads mailbox-shaped docs via ES _bulk into a throwaway node, flushes,
# settles, then reports VmRSS/VmHWM and the ingest-memory ledger's view
# (jemalloc allocated/active/resident). Variants separate per-doc from
# per-byte retention:
#   A  20k docs x 10 KB body   (~210 MB source)
#   B  20k docs x 100 B body   (~2 MB source)
#   C  100k docs x 100 B body  (~10 MB source, 5x docs of B)
set -euo pipefail
PORT=${PORT:-9341}
WORK=$(mktemp -d /tmp/948-atrest.XXXXXX)
VARIANT=${1:-A}
case "$VARIANT" in
  A) DOCS=20000; BODY_KB=10;;
  B) DOCS=20000; BODY_KB=0;;
  C) DOCS=100000; BODY_KB=0;;
  *) echo "variant must be A|B|C"; exit 2;;
esac
# BODY_KB=0 -> tiny bodies: patch via --body-kb 0 in the generator = 0*1024
# bytes target; body() would return empty string. Give it 100 B instead.
# Binary under test: default = the worktree build; override for A/B, e.g.
#   XERJ=/root/948-runs/bin/xerj-draft-citest ./atrest_bisect.sh A   # main
#   XERJ=/root/948-runs/bin/xerj-948-fixed    ./atrest_bisect.sh A   # fixed
XERJ=${XERJ:-/root/xerj-main/engine/target-b/ci-test/xerj}
HERE=$(cd "$(dirname "$0")" && pwd)
URL=http://127.0.0.1:$PORT

cat > "$WORK/node.toml" <<TOML
[server]
es_compat_port = $PORT
rest_port = $((PORT+1))
grpc_port = $((PORT+2))
data_dir = "$WORK/data"
[tls]
enabled = false
[embedding]
mode = "lexical"
TOML
mkdir -p "$WORK/data"

# No memory cap (large limit): the breaker must not 429 the load — we want
# the retention, not the pressure behaviour, in this experiment.
XERJ_MAX_PROCESS_MEMORY_MB=65536 \
XERJ_INGEST_MEMORY_TRACE=summary \
XERJ_INGEST_MEMORY_SAMPLE_MS=1000 \
XERJ_INGEST_MEMORY_OUTPUT="$WORK/ledger.ndjson" \
nohup "$XERJ" -c "$WORK/node.toml" --embed-mode lexical > "$WORK/server.log" 2>&1 &
SERVER=$!
cleanup() { kill "$SERVER" 2>/dev/null || true; wait "$SERVER" 2>/dev/null || true; }
trap cleanup EXIT

KEY=""
for _ in $(seq 1 120); do
  [ -s "$WORK/data/admin.key" ] && KEY=$(cat "$WORK/data/admin.key")
  [ -n "$KEY" ] && curl -fsS -H "Authorization: ApiKey $KEY" "$URL/_cluster/health" >/dev/null 2>&1 && break
  sleep 0.5
done
curl -fsS -H "Authorization: ApiKey $KEY" "$URL/_cluster/health" >/dev/null || { echo "node did not come up"; tail -5 "$WORK/server.log"; exit 1; }
LISTENER=$(ss -ltnp 2>/dev/null | grep -E ":$PORT\b" | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
[ "$LISTENER" = "$SERVER" ] || { echo "refusing: :$PORT held by pid ${LISTENER:-?} not $SERVER"; exit 2; }

rss() { awk '/^VmRSS:/{print $2}' "/proc/$SERVER/status"; }
hwm() { awk '/^VmHWM:/{print $2}' "/proc/$SERVER/status"; }
echo "variant=$VARIANT docs=$DOCS body_kb=$BODY_KB idle_rss_kb=$(rss) idle_hwm_kb=$(hwm)"

python3 "$HERE/synth_bulk_ingest.py" --url "$URL" --docs "$DOCS" --body-kb "$BODY_KB" \
    --bulk-mb 8 --key "$WORK/data/admin.key" --json > "$WORK/load.json" || {
  cat "$WORK/load.json"; tail -20 "$WORK/server.log"; exit 3; }
cat "$WORK/load.json"

# Explicit flush, then settle: merges (if any) finish, sampler records at-rest.
curl -fsS -X POST -H "Authorization: ApiKey $KEY" "$URL/bench-docs/_flush" >/dev/null
echo "post-flush rss_kb=$(rss)"
sleep 45
echo "settled rss_kb=$(rss) hwm_kb=$(hwm)"
echo "cache/view stats:"; curl -fsS -H "Authorization: ApiKey $KEY" "$URL/bench-docs/_stats" > "$WORK/stats.json" || true; wc -c < "$WORK/stats.json"
echo "index bytes on disk: $(du -sk "$WORK/data" | cut -f1) kB"

# Retention release test: does deleting the index return the pages?
BEFORE_DEL=$(rss)
curl -fsS -X DELETE -H "Authorization: ApiKey $KEY" "$URL/bench-docs" >/dev/null || true
sleep 10
echo "after-delete rss_kb=$(rss) (was $BEFORE_DEL before delete)"

# Ledger tail: last snapshot line (at-rest view).
echo "ledger last snapshot:"
tail -2 "$WORK/ledger.ndjson" | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); print(json.dumps(d.get("snapshot", d))[:1500])' || tail -2 "$WORK/ledger.ndjson"
echo "WORKDIR=$WORK"
