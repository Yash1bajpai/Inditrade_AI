"""
Module B: Hybrid Qualitative Legal & Policy Retriever (Dense Semantic + Sparse BM25 + RRF)

Queries the Qdrant vector database (`trade_policy_compliance`) and the serialized BM25 sparse index
(`bm25_index.pkl`), combining both through Reciprocal Rank Fusion (RRF) to retrieve the exact governing
DGFT export/import circulars, tariff schedules, and policy exemptions.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import joblib
import argparse
from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

DEFAULT_QDRANT_PATH = "data/cache/qdrant_index"
DEFAULT_BM25_PATH = "data/cache/bm25_index.pkl"
DEFAULT_COLLECTION = "trade_policy_compliance"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class HybridPolicyRetriever:
    """
    Hybrid Retriever combining Qdrant dense vector search with BM25 keyword search.
    Uses Reciprocal Rank Fusion (RRF) to merge ranks and ensure both semantic intent
    and exact keyword matches (like HS codes, circular numbers) are captured.
    """
    def __init__(
        self,
        qdrant_path: str = DEFAULT_QDRANT_PATH,
        bm25_path: str = DEFAULT_BM25_PATH,
        collection_name: str = DEFAULT_COLLECTION,
        model_name: str = DEFAULT_MODEL
    ):
        self.qdrant_path = qdrant_path
        self.bm25_path = bm25_path
        self.collection_name = collection_name
        self.model_name = model_name
        
        print(f"[*] Initializing HybridPolicyRetriever (Model: {model_name})...")
        self.embedder = SentenceTransformer(model_name)
        self.qdrant_client = QdrantClient(path=qdrant_path)
        
        print(f"[*] Loading serialized BM25 index from -> {bm25_path}")
        if not os.path.exists(bm25_path):
            raise FileNotFoundError(f"BM25 index not found at {bm25_path}. Run indexer.py first!")
        data = joblib.load(bm25_path)
        self.bm25 = data["bm25"]
        self.docs = data["docs"]
        self.doc_id_to_doc = {d["doc_id"]: d for d in self.docs}
        print(f"[OK] Hybrid Retriever ready! Total indexed legal documents: {len(self.docs)}")

    def search_dense(self, query: str, top_k: int = 15, doc_type_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Executes dense vector search on Qdrant.
        """
        query_vector = self.embedder.encode(query, normalize_embeddings=True).tolist()
        
        query_filter = None
        if doc_type_filter:
            query_filter = Filter(
                must=[FieldCondition(key="doc_type", match=MatchValue(value=doc_type_filter))]
            )
            
        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k
        ).points
        
        dense_hits = []
        for rank, r in enumerate(results):
            payload = r.payload or {}
            dense_hits.append({
                "doc_id": payload.get("doc_id", f"dense_{rank}"),
                "dense_rank": rank + 1,
                "dense_score": float(r.score),
                "payload": payload
            })
        return dense_hits

    def search_sparse(self, query: str, top_k: int = 15, doc_type_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Executes BM25 sparse keyword search across tokenized corpus.
        """
        tokens = [word.lower().strip(",.()[]{}\"'") for word in query.split() if len(word) > 1]
        if not tokens:
            return []
            
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1]
        
        sparse_hits = []
        rank = 1
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            doc = self.docs[idx]
            if doc_type_filter and doc["doc_type"] != doc_type_filter:
                continue
            sparse_hits.append({
                "doc_id": doc["doc_id"],
                "sparse_rank": rank,
                "sparse_score": float(scores[idx]),
                "payload": doc
            })
            rank += 1
            if rank > top_k:
                break
        return sparse_hits

    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        rrf_k: int = 60,
        doc_type_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Combines dense vector search and sparse BM25 search using Reciprocal Rank Fusion (RRF).
        RRF score = (alpha / (rrf_k + dense_rank)) + ((1 - alpha) / (rrf_k + sparse_rank))
        """
        dense_hits = self.search_dense(query, top_k=top_k * 3, doc_type_filter=doc_type_filter)
        sparse_hits = self.search_sparse(query, top_k=top_k * 3, doc_type_filter=doc_type_filter)
        
        rrf_scores = {}
        payload_map = {}
        ranks_map = {}
        
        # Process dense hits
        for h in dense_hits:
            doc_id = h["doc_id"]
            payload_map[doc_id] = h["payload"]
            ranks_map.setdefault(doc_id, {})["dense_rank"] = h["dense_rank"]
            ranks_map[doc_id]["dense_score"] = h["dense_score"]
            rrf_scores[doc_id] = alpha / (rrf_k + h["dense_rank"])
            
        # Process sparse hits
        for h in sparse_hits:
            doc_id = h["doc_id"]
            if doc_id not in payload_map:
                payload_map[doc_id] = h["payload"]
            ranks_map.setdefault(doc_id, {})["sparse_rank"] = h["sparse_rank"]
            ranks_map[doc_id]["sparse_score"] = h["sparse_score"]
            
            sparse_contrib = (1.0 - alpha) / (rrf_k + h["sparse_rank"])
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + sparse_contrib
            
        # Sort by combined RRF score descending
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)[:top_k]
        
        final_results = []
        for idx, doc_id in enumerate(sorted_doc_ids):
            payload = payload_map[doc_id]
            r_info = ranks_map.get(doc_id, {})
            final_results.append({
                "rank": idx + 1,
                "rrf_score": float(rrf_scores[doc_id]),
                "doc_id": doc_id,
                "doc_type": payload.get("doc_type", ""),
                "title": payload.get("title", ""),
                "notification_no": payload.get("notification_no", ""),
                "date": payload.get("date", ""),
                "ministry": payload.get("ministry", ""),
                "dense_rank": r_info.get("dense_rank", "N/A"),
                "sparse_rank": r_info.get("sparse_rank", "N/A"),
                "snippet": payload.get("text", "")[:450] + ("..." if len(payload.get("text", "")) > 450 else ""),
                "full_text": payload.get("text", "")
            })
            
        return final_results


def main():
    parser = argparse.ArgumentParser(description="Test Hybrid RAG Legal Retriever")
    parser.add_argument("--query", type=str, default="export restrictions or duty on sugar or electronics", help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to retrieve")
    parser.add_argument("--filter", type=str, default=None, choices=["DGFT_Policy", "DGFT_OCR", "PIB_Press_Release"], help="Filter by document type")
    args = parser.parse_args()
    
    retriever = HybridPolicyRetriever()
    print(f"\n[*] Executing Hybrid Search for Query: '{args.query}'\n")
    results = retriever.search(args.query, top_k=args.top_k, doc_type_filter=args.filter)
    
    for r in results:
        print(f"[{r['rank']}] [Score: {r['rrf_score']:.4f}] {r['doc_type']} | {r['title']}")
        print(f"    * Notification No: {r['notification_no']} | Date: {r['date']}")
        print(f"    * Ranks -> Dense: {r['dense_rank']} | Sparse (BM25): {r['sparse_rank']}")
        print(f"    * Snippet: {r['snippet']}\n" + "-" * 80)


if __name__ == "__main__":
    main()
