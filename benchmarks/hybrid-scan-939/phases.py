#!/usr/bin/env python3
"""Split traced semantic requests into phases, per latency.py arm.

    python3 phases.py server.log latency.log

Start the node with XERJ_TRACE_SEMANTIC_PHASES=1 NO_COLOR=1. latency.py prints
ARM_START/ARM_END stamps (UTC, 1 s resolution); this buckets the node's
`semantic_phase=` lines by those windows. One client, closed loop, so lines of
one request are contiguous: embed_start → embed_complete → start_brute →
load_segment* → scored → complete.
"""
import re
import sys
from datetime import datetime, timezone

server_log, latency_log = sys.argv[1], sys.argv[2]
arms = []
start = {}
for line in open(latency_log):
    m = re.match(r"ARM_(START|END) (.+) (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)$", line.strip())
    if not m:
        continue
    t = datetime.strptime(m.group(3), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    if m.group(1) == "START":
        start[m.group(2)] = t
    else:
        arms.append((m.group(2), start[m.group(2)], t))

ts_re = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)\.\d+Z")
kv = lambda line, key: (lambda m: int(m.group(1)) if m else None)(re.search(key + r"=(\d+)", line))
requests, cur = [], None
for line in open(server_log, errors="replace"):
    if "semantic_phase=" not in line:
        continue
    m = ts_re.match(line)
    if not m:
        continue
    t = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    if "semantic_phase=embed_start" in line:
        cur = {"t": t}
        requests.append(cur)
    elif cur is None:
        continue
    elif "semantic_phase=embed_complete" in line:
        cur["embed"] = kv(line, "elapsed_ms")
    elif "semantic_phase=scored" in line:
        cur["collect"] = kv(line, "collect_ms")
        cur["scored_at"] = kv(line, "elapsed_ms")
        cur["scored_docs"] = kv(line, "scored")
    elif "semantic_phase=complete" in line:
        cur["total"] = kv(line, "elapsed_ms")


def p50(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else float("nan")


for name, t0, t1 in arms:
    rs = [r for r in requests if t0 <= r["t"] <= t1 and "total" in r]
    if not rs:
        continue
    score = [r["scored_at"] - r["collect"] for r in rs if r.get("scored_at") is not None and r.get("collect") is not None]
    tail = [r["total"] - r["scored_at"] for r in rs if r.get("scored_at") is not None]
    print(f"{name:34s} n={len(rs):4d} embed p50={p50([r.get('embed') for r in rs]):6.1f}"
          f"  collect p50={p50([r.get('collect') for r in rs]):6.1f}  score p50={p50(score):6.1f}"
          f"  topk+hits p50={p50(tail):6.1f}  scan_total p50={p50([r['total'] for r in rs]):6.1f}"
          f"  scored_docs p50={p50([r.get('scored_docs') for r in rs])}")
