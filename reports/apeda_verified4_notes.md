# Bounded APEDA addition, 20 October 2026

User approved four page-checked DGFT notifications via the delayed APEDA mirror.
Only 41, 36, 34, 35 of FY 2026-27 are included. Twenty other candidate gaps remain
quarantined. This does not cover DGFT Public Notices or establish current rules.

The four one-page documents are kept as complete parent payloads so exceptions
and operative clauses are not split away from the answer context. All 5,195 old
payloads remain byte-for-byte equivalent as Python data. Total: 5,199 payloads.

Validation before promotion used a separate Qdrant collection cloned from all
5,195 production vectors/payloads, plus the four complete checked documents;
cloud embeddings used the production intfloat/multilingual-e5-small model.
Sparse retrieval used the rebuilt complete corpus and the application's real
BM25 tokenizer, then real backend _rrf_fuse (8 dense + 8 sparse -> 5 contexts).
Sixteen dated/identifier/subject queries (four per new document) each included the
intended new document in the five contexts. Not always rank one; older related
notifications still appear. Twenty before/after old-subject queries retained all
five original parents for nineteen queries; one retained four and added the new
RoDTEP document. This measures retrieval stability, not legal answer correctness.

Six TestClient router calls used real remote candidate dense retrieval and real
full-corpus sparse retrieval. Generation was replaced with the existing local
extractive fallback, so no live LLM-answer or provider-availability claim is made.
The four subject calls included mirror/date/coverage labels in citations; current
policy and fabricated-tax prompts were refused. Reports contain full results.

The audit corrections were compared with page pixels twice by the SAME agent,
not independent or human legal review. Audited text hashes and PDF hashes are in
data/reviewed/apeda_vetted.json. Only exact approved text is admitted by the build
script; unreviewed/altered candidates are rejected.

Rollback: revert this bounded index addition to aa7c9b2, and remove only the four
new deterministic point IDs from trade_policy_e5_small_v1. Existing points were
not changed. No release assets, model weights, or historical corpus were replaced.
Render /health returned HTTP503 Service Suspended during validation, so the live
app's use of the updated artifact cannot yet be verified.
