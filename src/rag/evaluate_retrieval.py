"""
Module B: Retrieval Quality Evaluation Harness (hit@k / MRR)

Measures how well the policy retriever surfaces the exact source document for a
known question. Gold labels come from `policy_qa_dataset.jsonl`: every QA pair
was generated from a specific policy chunk (`context_snippet`), so a retrieval
is correct when one of the returned documents actually contains that snippet.

OCR text is noisy ("lndia" for "India"), so gold matching uses content-token
containment rather than exact substring: a hit counts when at least
GOLD_OVERLAP_THRESHOLD of the gold snippet's content tokens appear in the
retrieved text.

Usage:
    python -m src.rag.evaluate_retrieval --mode hybrid --max-questions 300
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import argparse
import json
import re
import time
from typing import List, Dict, Any

GOLD_OVERLAP_THRESHOLD = 0.6
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Boilerplate that appears on nearly every DGFT gazette page and carries no
# discriminative signal for judging whether the right document was retrieved.
_STOPWORDS = {
    "the", "of", "to", "in", "and", "or", "for", "is", "are", "be", "by", "on",
    "a", "an", "as", "at", "it", "its", "with", "under", "section", "sub",
    "part", "ii", "iii", "government", "india", "ministry", "commerce",
    "industry", "directorate", "general", "foreign", "trade", "regulation",
    "development", "act", "notification", "no", "new", "delhi", "subject",
    "regarding", "sourced", "gazette", "extraordinary", "published", "ll",
    "sub", "e", "s", "d", "b", "c", "f", "g",
}


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _content_tokens(text: str) -> set:
    return {t for t in _tokens(text) if len(t) > 2 and t not in _STOPWORDS}


def load_eval_questions(qa_path: str, max_questions: int) -> List[Dict[str, Any]]:
    questions = []
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            snippet = obj.get("context_snippet", "")
            question = obj.get("question", "")
            if len(question) >= 10 and _content_tokens(snippet):
                questions.append({
                    "qa_id": obj.get("qa_id", ""),
                    "question": question,
                    "gold_snippet": snippet,
                    "gold_tokens": _content_tokens(snippet),
                    "doc_type": obj.get("doc_type", ""),
                })
            if max_questions and len(questions) >= max_questions:
                break
    return questions


def is_relevant(retrieved_text: str, gold_tokens: set, threshold: float = GOLD_OVERLAP_THRESHOLD) -> bool:
    if not gold_tokens:
        return False
    hit_tokens = _content_tokens(retrieved_text)
    overlap = len(gold_tokens & hit_tokens) / len(gold_tokens)
    return overlap >= threshold


def build_parent_texts() -> Dict[str, str]:
    """Map parent_doc_id -> full document text (chunks joined in order).

    Judging happens at the parent level so chunked and whole-doc indexes are
    measured on equal footing: a chunked index is correct when it surfaces a
    chunk of the right document, not only when one chunk contains the entire
    gold snippet.
    """
    from src.rag.sparse_index import ProdBM25Index

    index = ProdBM25Index.load()
    parts: Dict[str, Dict[int, str]] = {}
    for doc in index.docs:
        parent = doc.get("parent_doc_id") or doc.get("doc_id")
        parts.setdefault(parent, {})[doc.get("chunk_index", 0)] = doc.get("text", "")
    return {
        parent: "\n".join(text for _, text in sorted(chunks.items()))
        for parent, chunks in parts.items()
    }


def evaluate(results: List[List[Dict[str, Any]]], eval_set: List[Dict[str, Any]], k_values=(1, 3, 5, 10)) -> Dict[str, Any]:
    parent_texts = build_parent_texts()
    doc_hits_at = {k: 0 for k in k_values}
    chunk_hits_at = {k: 0 for k in k_values}
    doc_rr = 0.0
    chunk_rr = 0.0
    judged = 0

    for retrieved, item in zip(results, eval_set):
        gold = item["gold_tokens"]
        if not gold:
            continue
        judged += 1
        first_doc_rank = None
        first_chunk_rank = None
        for rank, doc in enumerate(retrieved, start=1):
            text = doc.get("full_text") or doc.get("text") or ""
            parent = doc.get("parent_doc_id") or doc.get("doc_id")
            parent_full = parent_texts.get(parent, "")
            if first_doc_rank is None and is_relevant(parent_full or text, gold):
                first_doc_rank = rank
            if first_chunk_rank is None and is_relevant(text, gold, threshold=0.35):
                first_chunk_rank = rank
            if first_doc_rank is not None and first_chunk_rank is not None:
                break
        for k in k_values:
            if first_doc_rank is not None and first_doc_rank <= k:
                doc_hits_at[k] += 1
            if first_chunk_rank is not None and first_chunk_rank <= k:
                chunk_hits_at[k] += 1
        if first_doc_rank is not None:
            doc_rr += 1.0 / first_doc_rank
        if first_chunk_rank is not None:
            chunk_rr += 1.0 / first_chunk_rank

    metrics = {"judged": judged}
    for k in k_values:
        if judged:
            metrics[f"doc_hit@{k}"] = round(doc_hits_at[k] / judged, 4)
            metrics[f"chunk_hit@{k}"] = round(chunk_hits_at[k] / judged, 4)
    metrics["doc_mrr@10"] = round(doc_rr / judged, 4) if judged else 0.0
    metrics["chunk_mrr@10"] = round(chunk_rr / judged, 4) if judged else 0.0
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate policy retrieval quality")
    parser.add_argument("--mode", choices=["dense", "hybrid"], default="hybrid")
    parser.add_argument("--max-questions", type=int, default=300)
    parser.add_argument("--qa-path", type=str, default="data/processed/policy_qa_dataset.jsonl")
    parser.add_argument("--output", type=str, default=None, help="Optional path to write the metrics JSON")
    args = parser.parse_args()

    from src.rag.retriever import HybridPolicyRetriever

    eval_set = load_eval_questions(args.qa_path, args.max_questions)
    print(f"[*] Evaluating {len(eval_set)} questions in '{args.mode}' mode...")

    retriever = HybridPolicyRetriever()
    results = []
    t0 = time.time()
    for i, item in enumerate(eval_set):
        if args.mode == "dense":
            docs = retriever.search_dense(item["question"], top_k=10)
            # dense hits carry the payload directly
            normalized = [{**h["payload"], "full_text": h["payload"].get("text", "")} for h in docs]
        else:
            normalized = retriever.search(item["question"], top_k=10)
        results.append(normalized)
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(eval_set)} queried ({time.time() - t0:.0f}s)")

    metrics = evaluate(results, eval_set)
    metrics["mode"] = args.mode
    metrics["questions"] = len(eval_set)
    metrics["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    print("\n=== RETRIEVAL QUALITY ===")
    for key, val in metrics.items():
        print(f"  {key}: {val}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[Exported] {args.output}")


if __name__ == "__main__":
    main()
