"""THE ACCEPTANCE GATE: pip-installed jev-reranker, UNMODIFIED, pointed at a
XERJ node.

    TYPESAFE_ENDPOINT=http://localhost:<native-rest-port>/v1/systemone
    TYPESAFE_API_KEY=<any non-empty string — the node ignores it, localhost only>

Ranks the fixed gate documents (gate-docs.json: 5 spam, 5 ham) with the
client's own listwise pipeline and asserts two things:

1. WIRE: the call completes with no ResponseValidationError — answers keyed
   exactly, noul numeric in 0..1, model echoed as a non-empty string, usage
   non-negative ints. The client enforces this; surviving it IS the gate.
2. SEMANTICS: the local vote separates the classes — mean noul over spam
   documents strictly exceeds mean noul over ham documents by >= 0.3.

Run: python3 gate_run.py [native_url_base]
"""
import json, os, sys

docs = json.load(open("gate-docs.json"))
texts = [d["text"] for d in docs]
labels = [d["label"] for d in docs]
# The query rides in state.query and joins every vote text, so its terms must
# be NEUTRAL for the history: a spammy query ("winner claim prize") retrieves
# spam neighbours for every document and the vote says nothing about the doc.
# Every term here is absent from the SMS corpus (checked against sms.tsv), so
# the document drives the vote. That asymmetry — query vocabulary pollutes a
# classification history — is documented in docs/RERANK.md.
QUERY = "triage inbox unsolicited correspondence"

from jev_reranker import JevReranker

rr = JevReranker()  # reads TYPESAFE_ENDPOINT / TYPESAFE_API_KEY from the env
result = rr.rerank(QUERY, texts, detail=True)

# {"results": [{"document_index", "score", "text", ...}, ...]} — order follows
# the client's ranking; re-key by the document index it reports.
by_index = {r["document_index"]: r["score"] for r in result["results"]}
scores = [by_index[i] for i in range(len(texts))]
print("client resolved model:", result["detail"]["resolved_models"])
for lab, txt, s in zip(labels, texts, scores):
    print(f"  {s:.4f}  {lab:4}  {txt[:70]}")

assert len(scores) == len(texts)
assert all(isinstance(s, float) and 0.0 <= s <= 1.0 for s in scores), scores
spam = [s for s, l in zip(scores, labels) if l == "spam"]
ham = [s for s, l in zip(scores, labels) if l == "ham"]
ms, mh = sum(spam) / len(spam), sum(ham) / len(ham)
print(f"mean noul  spam={ms:.4f}  ham={mh:.4f}  gap={ms - mh:.4f}")
assert ms - mh >= 0.3, f"vote did not separate: {ms:.3f} vs {mh:.3f}"
print("GATE PASSED: unmodified jev-reranker ranked from the XERJ node")
