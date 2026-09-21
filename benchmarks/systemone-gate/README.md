# The /v1/systemone acceptance gate — jev-reranker, unmodified, on a XERJ node

Run 2026-09-20, locally built `xerj-server` at the wire-compat commit,
`--insecure --port 9440` (native REST on 9441), throwaway data dir,
`[decisions] index = "sms"` (`gate-decisions.toml`: k=10, text/label fields,
`positive_label = "spam"`). Client: `pip install jev-reranker` (0.1.0, MIT) in a
clean venv — no patches, no subclassing, its own listwise instruction template,
its own splitting, its own strict response validation. The only configuration:

```sh
TYPESAFE_ENDPOINT=http://localhost:9441/v1/systemone
TYPESAFE_API_KEY=<any non-empty string>   # the node ignores it; localhost only
```

`TYPESAFE_BASE_URL` is silently ignored by the client — the endpoint variable
must carry the full path. (Verified in the wheel: `reranker.py:238`.)

## What ran

`gate_load_sms.py` indexes the 4,000-message SMS train split
(`benchmarks/decisions-as-retrieval`'s split: shuffle seed 7, first 4,000) and
fixes 10 held-out gate documents (5 spam, 5 ham, 40–140 chars) in
`gate-docs.json`. `gate_run.py` ranks them with `JevReranker().rerank(...)`.
The query is `triage inbox unsolicited correspondence` — every term verified
absent from the corpus, because `state.query` joins every vote text and a
vocabulary-bearing query ("winner claim prize") retrieves spam neighbours for
EVERY document, which measures the query, not the documents. First gate
attempt failed exactly there (spam 1.000 vs ham 0.785); the transcript below
is the fixed run.

## Result (`gate-transcript.txt`)

The client's own validation IS the wire gate: answers keyed exactly by the
sent question ids, every answer `{type: "noul", noul}` numeric in 0..1,
`model` a non-empty string (it resolved `xerj-history-vote-1` — the node's
truthful echo, never a Jev name), `usage` non-negative integers. The semantic
gate: mean noul 0.9014 over spam vs 0.0956 over ham (gap 0.806, threshold
0.3). No request left the node: the vote is an ordinary search of the `sms`
index; the module adds no outbound client.

## The response the client does not show

`/_decide` exposes the same vote with its evidence — neighbours, labels,
engine scores, weights, and an abstain verdict below `decisions.min_confidence`:

```sh
curl -s localhost:9440/_decide -H 'content-type: application/json' \
  -d '{"index":"sms","question":"URGENT! Your Mobile number has been awarded"}'
# {"index":"sms","k":10,"label":"spam","confidence":1.0,"abstain":false,
#  "neighbours":[{"_id":"1805","label":"spam",...,"weight":1.0}, ...]}
```

Zero-support questions are a 422 naming the question ids, never a fabricated
0.5; `score` questions (no vote analogue) are a 422 naming the alternative.
Both behaviours are pinned by `engine/crates/xerj-api/tests/systemone_http.rs`.
