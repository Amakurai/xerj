#!/usr/bin/env bash
# Console end-to-end suite against a real node. Usage: scripts/e2e.sh [fixture-dir]
# Builds xerj-server (scoped release), boots a throwaway open-mode node on a
# private port, indexes the fixture, runs Playwright. Never touches :9200.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURE="${1:-${XERJ_E2E_FIXTURE:-}}"
if [ -z "$FIXTURE" ]; then
  FIXTURE="$(mktemp -d)/takeout"
  python3 "$HERE/scripts/synthetic-takeout.py" --out "$FIXTURE" --messages 120 --seed 1
fi
export XERJ_E2E_FIXTURE="$FIXTURE"
export XERJ_PORT="${XERJ_PORT:-9377}"
( cd "$HERE/engine" && cargo build --release -j "${JOBS:-32}" -p xerj-server )
export XERJ_BIN="$HERE/engine/target/release/xerj"
cd "$HERE/xerj-ux/e2e"
[ -d node_modules/@playwright ] || npm ci --no-audit --no-fund
npx playwright install chromium >/dev/null
npx playwright test "$@"
