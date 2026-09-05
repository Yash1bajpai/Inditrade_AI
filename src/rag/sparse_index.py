"""
Production sparse retrieval: BM25 over the chunked policy corpus.

Loads the lzma-compressed BM25 artifact tracked in git (data/cache/
bm25_index.pkl.lzma, ~2.6MB) so the API always has a real keyword retriever —
exact HS-code / notification-number lookups — even when no Qdrant cluster is
reachable. No sentence-transformers, no network: safe for the 512MB Render
container.

The same class backs two consumers:
  - src/backend/database.py: as the no-Qdrant fallback retriever
  - src/backend/api/query.py: as the sparse half of RRF fusion
"""

import os
import pickle
import lzma
import numpy as np

DEFAULT_LZMA_PATH = "data/cache/bm25_index.pkl.lzma"
DEFAULT_PKL_PATH = "data/cache/bm25_index.pkl"


def tokenize_for_bm25(text: str) -> list:
    """Must stay token-compatible with indexer.py's corpus tokenization."""
    return [w.lower().strip(",.()[]{}\"'") for w in text.split() if len(w) > 1]


class _Hit:
    """Minimal ScoredPoint stand-in so callers can use .payload / .score."""

    def __init__(self, payload: dict, score: float, idx: int):
        self.id = idx
        self.score = float(score)
        self.payload = payload


class ProdBM25Index:
    def __init__(self, bm25, docs: list):
        self.bm25 = bm25
        self.docs = docs

    @classmethod
    def load(cls, lzma_path: str = DEFAULT_LZMA_PATH, pkl_path: str = DEFAULT_PKL_PATH) -> "ProdBM25Index":
        """Load the compressed artifact, falling back to a raw pickle if present."""
        if os.path.exists(lzma_path):
            with lzma.open(lzma_path, "rb") as f:
                data = pickle.load(f)
        elif os.path.exists(pkl_path):
            import joblib
            data = joblib.load(pkl_path)
        else:
            raise FileNotFoundError(
                f"BM25 artifact not found ({lzma_path} / {pkl_path}). Run src/rag/indexer.py first."
            )
        return cls(data["bm25"], data["docs"])

    def search(self, query_text: str, limit: int = 8, collection_name: str = None,
               doc_type_filter: str | None = None, **kwargs) -> list:
        # collection_name is accepted (and ignored) so this class can stand in
        # for a QdrantClient behind the same .search() call site.
        tokens = tokenize_for_bm25(query_text)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        order = np.argsort(scores)[::-1]

        hits, rank = [], 1
        for idx in order:
            if scores[idx] <= 0:
                break
            doc = self.docs[idx]
            if doc_type_filter and doc.get("doc_type") != doc_type_filter:
                continue
            hits.append(_Hit(doc, scores[idx], int(idx)))
            rank += 1
            if rank > limit:
                break
        return hits
