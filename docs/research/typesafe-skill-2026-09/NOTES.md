# Notes — measuring the Jev model against XERJ, then speaking its wire

Research log, 2026-09-19..20. Two questions, in the order we asked them:

1. Does the Jev judge model, called through its documented wire, actually
   improve XERJ's ranking — and what are its probabilities worth?
2. Can a XERJ node BE a System One endpoint — answer the same wire with a local
   vote, so a client written for TypeSafe's API works unmodified?

Raw artifacts: `benchmarks/beir-hybrid/results/2026-09-20-rerank-pilot/`
(pilot, 40 queries) and `.../2026-09-20-rerank-full/` (full run, 3 repeats,
calibration). This file is the narrative; the JSONs are the evidence.

## 1. The stage shipped unmeasured, and measured badly (0.3822)

docs/RERANK.md carried "Not verified: ranking quality with the real model" —
with a provider key we ran it: **0.3822 nDCG@10** against BM25's **0.7750**
over the first 40 judged SciFact queries. The stage made results worse.

Root cause, isolated at the raw API (not guessed): `build_jev_request` put
every candidate in shared `state` and asked each key an untargeted "Does this
document contain information that answers the query?" — so "this document"
resolved to nothing and the model judged the *pile* once, echoing ~that
probability per question. Within every window the probabilities were
near-uniform (spread ≈ 0.02 over 30 documents), which is query-level noise, not
ranking.

Two raw-API probes confirmed it (V1/V2, same shortlists, same day):

| Shape | nDCG@10 (40 q) |
|---|---:|
| V1 — candidates in `state`, untargeted question | 0.3822 |
| V2 — one noul per document, document inside its own `instructions`, question naming `` `document` `` | 0.8389 |

The fix (commit 7c6f496e) moved the stage to the V2 shape: 0.3822 → **0.8299**
through the stage, within provider wobble of the raw API's 0.8389. The wire
reference for the shape is `jev-reranker`'s `_payload()` — listwise mode,
`instructions.format(document="`documents.doc_0`")` — read from the installed
wheel, not from memory.

## 2. The full run: ranking up, calibration not there

300 SciFact + 323 NFCorpus queries, BM25 top-30 shortlists, 3 repeats with
shuffled query order, per-query medians (`raw_bench_full.py` in the results
dir):

| Dataset | BM25 | Jev median | W / L / T | cost |
|---|---:|---:|---|---:|
| SciFact | 0.6572 | 0.7410 | 82 / 30 / 188 | $0.4027 |
| NFCorpus | 0.3016 | 0.3312 | 118 / 74 / 131 | $0.3200 |

Findings that matter more than the headline:

- **Provider non-determinism is real.** Mean per-document |Δ| 0.0077–0.0096
  across repeats, max 0.08. Repeat-level means moved ±0.003 nDCG. Comparisons
  of Jev numbers tighter than ~±0.01 nDCG@10 are noise.
- **The probabilities are not calibrated to BEIR relevance.** ECE 0.1023 /
  0.1332; monotone curve (it discriminates) but bin 8 on SciFact is 0.85 mean
  confidence / 0.42 actual relevance. A Jev probability orders documents well
  and thresholds them badly against graded human labels. `min_score` on the
  raw number would cut documents qrels call relevant.
- The losses concentrate where BM25 already failed (empty/near-empty
  shortlists) and on NFCorpus's medical shorthand, where 30-candidate windows
  are mostly noise to a general-purpose judge.

## 3. Speaking the wire: /v1/systemone + /_decide

The second question inverted the first: TypeSafe's System One API is a
documented interface for typed judgement — `POST /v1/systemone` with
`{state, model, questions}`, answers typed `noul`/`choice`. XERJ already had
the retrieval answer for the same shape: index labelled history, answer a new
question by a rank-weighted vote over its k nearest neighbours, report the
winning label's share as the probability (benchmarks/decisions-as-retrieval:
Banking77 0.933 accuracy / ECE 0.012, SMS 0.983, sub-millisecond, BM25-only).

So the node now speaks both surfaces (this branch):

- `POST /v1/systemone` (native router) — the documented request/response shape,
  answers from the vote, so `jev-reranker` unmodified with
  `TYPESAFE_ENDPOINT=http://localhost:<port+1>/v1/systemone` reranks off the
  node. Acceptance gate: the pip-installed client, 5 spam / 5 ham held-out SMS
  documents, spam-vs-ham mean-noul separation — run 2026-09-20, see §4.
- `POST /_decide` (ES-compat router) — the same vote without the wire costume:
  per-request index/k/positive_label, the neighbours with labels, scores and
  weights in the response, abstain below `decisions.min_confidence`.

Nothing leaves the node: the vote is an ordinary search against an ordinary
index; the module adds no outbound client. That is the point — same interface,
evidence stays home.

**Two deliberate wire breaks, both tested so they cannot regress:**

1. `model` is echoed as `xerj-history-vote-1`, never as a Jev model name.
    Echoing `jev-1.13.0` would claim the hosted model produced these
    probabilities. The requested name is reported at
    `decisions.requested_model`. (The client validates `model` is a non-empty
    string — any truthful id passes.)
2. Zero support is a 422 naming the question ids, never a fabricated 0.5. A
    question whose vote finds no labelled neighbour carries no information;
    inventing a probability is the silent-fake defect class. `/_decide` says
    the same thing as `abstain` + `reason`.

`score` questions (2–10 ordinal levels) have no vote analogue and no
benchmark: 422 with the alternative named.

Design notes from the client source (wheel, MIT — read before writing the
endpoint): `instructions` is polymorphic (string | object | array); backtick
paths (`documents.doc_0`) resolve against `state`; `criteria` rides on every
question even for nouls; answers must be keyed EXACTLY by the sent ids with
only documented fields inside each answer (extras are safe at the top level);
`usage` must be JSON ints; retries on {429, 500, 502, 503, 504, 529}×8, no
redirects. All of this is asserted in `engine/crates/xerj-api/tests/
systemone_http.rs`.

## 4. The gate run (2026-09-20)

Node: locally built binary, `--insecure --port 9440`, throwaway data dir,
`[decisions] index = "sms"` (4,000 SMS train messages, labels ham/spam,
`positive_label = "spam"`, k = 10). Client: `pip install jev-reranker` in a
clean venv, `TYPESAFE_ENDPOINT=http://localhost:9441/v1/systemone`,
`TYPESAFE_API_KEY=<dummy>` — unmodified, listwise mode, its own instruction
template, its own splitting and validation. Documents: 5 spam + 5 ham
held-out messages, fixed in `gate-docs.json`. The client's strict response
validation IS the wire gate; the mean-noul separation is the semantic gate.
(Spam-vs-ham was chosen because it is the noul shape natively: labels are
literally yes/no.)

## What we claim, and what we do not

- Measured, ours: everything in the tables above, on our node, our shortlists,
  the pinned model, with the runner in the results dir.
- Not claimed: anything about Jev on Banking77/SMS (TypeSafe has published no
  such numbers; we did not run it there); any zero-shot capability for the
  local vote (it needs labelled history — where there is none, a judge model
  is the right tool); "1.00 recall", TB-scale, neural-default — none of it.
- The ECE contrast (Jev 0.10–0.13 cross-domain vs vote 0.012 in-domain) is a
  contrast of *targets*, not a scoreboard: calibration is a property of the
  task's labels, and both numbers are honest on their own task.
