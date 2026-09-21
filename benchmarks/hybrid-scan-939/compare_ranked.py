#!/usr/bin/env python3
"""Compare the ranked `(_id, _score)` lists two latency.py runs wrote. Exit 1 on ANY difference.

    python3 compare_ranked.py before.ranked.json after.ranked.json

Zero tolerance: same ids, same order, same `_score` JSON number. The last arm
(one request repeated) is compared too; it is served from the result cache on
both sides and must still agree.
"""
import json
import sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
bad = 0
if a.keys() != b.keys():
    print("ARMS DIFFER:", sorted(a.keys() ^ b.keys()))
    bad += 1
for arm in a:
    if arm not in b:
        continue
    ra, rb = a[arm], b[arm]
    if len(ra) != len(rb):
        print(f"{arm}: query count differs {len(ra)} vs {len(rb)}")
        bad += 1
        continue
    diffs = sum(1 for x, y in zip(ra, rb) if x != y)
    hits = sum(len(x) for x in ra)
    print(f"{arm:34s} queries={len(ra)} hits_compared={hits} differing_queries={diffs}")
    bad += diffs
print("IDENTICAL" if bad == 0 else f"NOT IDENTICAL: {bad} differences")
sys.exit(1 if bad else 0)
