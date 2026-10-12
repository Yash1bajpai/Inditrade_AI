# Free cloud RAG restoration (Oct 12, 2026)

## Prepared, not deployed

- Keep `trade_policy_compliance` (BGE) unchanged: 5,195 points.
- New `trade_policy_e5_small_v1`: 5,195 points, 384-d cosine, free `intfloat/multilingual-e5-small` cloud inference.
- Full payload text/source-ID equality verified across both collections. E5 uses the provider's automatic query/passage prefixes and 512-token model context. Long original chunks remain in payloads but inference can truncate their vectors.
- Activate only after review: Render `QDRANT_EMBEDDING_MODE=cloud_e5`, then deploy this branch's code. Without this variable, the existing BGE path remains selected. Roll back by reverting the code and removing the variable; no data deletion is needed.
- Require qdrant-client >=1.19.1 for Document/cloud inference. No model downloads on Render.

## Supabase

Existing free project was paused. Dashboard restore retained the data and fixed DNS. An authenticated read of `chat_history` using Render's existing publishable key returned 200 with an existing row. No settings/key changes or database writes were made during this check. Insert permissions and new chat persistence still need a reviewed production test. Code no longer reports mock persistence as success; `history_saved` reflects insert outcome.

## Evidence safety and limits

Both generation paths now request exact quotes, not policy explanations from model knowledge. Code accepts only verbatim spans of retrieved text. Invented statements, altered numbers, malformed responses and questions explicitly asking for current/live rules abstain. Old excerpts are labelled as historical reference, not current policy advice. `grounded` means a returned quote passed this check; it is not a guarantee of topical relevance, source truth or present applicability. Citation strings list retrieved documents, not independent confirmation of every quote.

Live probes were small and are not a benchmark: six dense probes yielded one exact quote and five abstentions before adding the current-rule guard; five full hybrid probes then all abstained. A sparse-only Pharma Grade Sugar probe returned exact historical conditions. This is an evidence-safety improvement, not demonstrated broad answer helpfulness. A deterministic verbatim matching-passage fallback then returned relevant historical excerpts for 2/5 probes (sugar and Advance Authorisation); current duty, current rice permission and invented duty still abstained, 3/5. Those checks used live hybrid retrieval with forced malformed generation to exercise the same fallback. Excerpts are labelled as partial reference, not a full explanation. No unsupported summary is generated. Further retrieval/context selection work is needed before claiming quality is fully fixed.

Supabase infrastructure is repaired. Production code/embedding-mode switch remain pending review. Forecast/model files were not changed. No paid inference or upgrade was used.

## Tests

Run `pytest tests/`. Unscoped pytest also collects two existing CLI inference scripts under `src/models/` whose function parameters are not pytest fixtures; those are not part of the backend unit suite.
