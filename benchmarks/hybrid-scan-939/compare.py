#!/usr/bin/env python3
"""Compare two identity.py dumps. Exit 1 on ANY difference.

    python3 compare.py before.json after.json

Tolerance is ZERO: same ids, same order, same `_score` (the exact JSON number),
same `_source`, same totals and relations, same passages. The scan rewrite in
#939 is meant to be bit-identical, so a tolerance would only hide a defect.
"""
import json
import sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
bad = 0
if a.keys() != b.keys():
    print("ARMS DIFFER:", sorted(a.keys() ^ b.keys()))
    bad += 1
for arm in sorted(a.keys() & b.keys()):
    rows_a, rows_b = a[arm], b[arm]
    diffs, hits = 0, 0
    if len(rows_a) != len(rows_b):
        print(f"{arm}: query count differs {len(rows_a)} vs {len(rows_b)}")
        bad += 1
        continue
    for ra, rb in zip(rows_a, rows_b):
        hits += len(ra["response"].get("hits", {}).get("hits", []))
        if ra != rb:
            diffs += 1
            if diffs <= 3:
                ha = [(h["_id"], h.get("_score")) for h in ra["response"].get("hits", {}).get("hits", [])]
                hb = [(h["_id"], h.get("_score")) for h in rb["response"].get("hits", {}).get("hits", [])]
                first = next((i for i, (x, y) in enumerate(zip(ha, hb)) if x != y), None)
                print(f"  {arm}: DIFF q={ra['q'][:60]!r} first_differing_rank={first}"
                      f" before={ha[first] if first is not None else None} after={hb[first] if first is not None else None}"
                      f" total {ra['response'].get('hits', {}).get('total')} vs {rb['response'].get('hits', {}).get('total')}")
    print(f"{arm:32s} queries={len(rows_a)} hits_compared={hits} differing_responses={diffs}")
    bad += diffs
print("IDENTICAL" if bad == 0 else f"NOT IDENTICAL: {bad} differing responses")
sys.exit(0 if bad == 0 else 1)
