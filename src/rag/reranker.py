"""
Module B: Cross-Encoder Legal Policy Reranker & Compliance Filter

Takes the candidate documents retrieved by `retriever.py` and reranks them using a high-precision
Cross-Encoder model (`BAAI/bge-reranker-base` or exact keyword relevance fallback) to filter out
tangential administrative circulars and isolate exact legal clauses and tariff schedules.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from typing import List, Dict, Any, Optional

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


class PolicyReranker:
    """
    Reranks retrieved qualitative policy hits to maximize legal citation precision
    and eliminate irrelevant background circulars.
    """
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, use_gpu: bool = False):
        self.model_name = model_name
        self.cross_encoder = None
        
        if CROSS_ENCODER_AVAILABLE:
            try:
                print(f"[*] Loading Cross-Encoder reranker model: {model_name}...")
                self.cross_encoder = CrossEncoder(model_name, max_length=512)
                print("[OK] Cross-Encoder loaded successfully!")
            except Exception as e:
                print(f"  [Warning] Could not initialize CrossEncoder {model_name} ({e}). Using exact relevance scoring.")
        else:
            print("  [Warning] `sentence_transformers.CrossEncoder` not imported. Using exact keyword overlap reranking.")

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Reranks candidate dictionary items against the input query.
        """
        if not candidates:
            return []
            
        if self.cross_encoder is not None:
            # Prepare pairs for Cross-Encoder
            pairs = [[query, doc["full_text"][:1000]] for doc in candidates]
            scores = self.cross_encoder.predict(pairs)
            
            for idx, score in enumerate(scores):
                candidates[idx]["rerank_score"] = float(score)
                candidates[idx]["original_rrf_score"] = candidates[idx].get("rrf_score", 0.0)
                
            # Sort descending by Cross-Encoder score
            reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
        else:
            # Fallback: Weighted overlap scoring plus RRF rank preservation
            query_terms = set(query.lower().split())
            for doc in candidates:
                text_lower = doc["full_text"].lower()
                overlap = sum(1 for t in query_terms if t in text_lower)
                # Boost if exact phrase in title or subject
                title_boost = 2.0 if any(t in doc.get("title", "").lower() for t in query_terms) else 0.0
                doc["rerank_score"] = float(doc.get("rrf_score", 0.0) * 10 + overlap + title_boost)
                doc["original_rrf_score"] = doc.get("rrf_score", 0.0)
                
            reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
            
        # Assign new clean ranks
        for rank, item in enumerate(reranked):
            item["final_rank"] = rank + 1
            
        return reranked


def main():
    from src.rag.retriever import HybridPolicyRetriever
    
    query = "export duty on sugar and molasses"
    print(f"\n=== END-TO-END QUALITATIVE RAG PIPELINE (RETRIEVER + RERANKER) ===")
    print(f"Query: '{query}'\n")
    
    retriever = HybridPolicyRetriever()
    candidates = retriever.search(query, top_k=10)
    print(f"[*] Retrieved {len(candidates)} candidate documents from Qdrant + BM25...")
    
    reranker = PolicyReranker()
    final_hits = reranker.rerank(query, candidates, top_k=3)
    
    print(f"\n=== TOP 3 VERIFIED LEGAL POLICY CITATIONS ===")
    for r in final_hits:
        print(f"[{r['final_rank']}] [Rerank Score: {r['rerank_score']:.4f} | RRF Score: {r['original_rrf_score']:.4f}]")
        print(f"    * Document Type : {r['doc_type']}")
        print(f"    * Title / Subject : {r['title']}")
        print(f"    * Notification #  : {r['notification_no']} | Date: {r['date']}")
        print(f"    * Legal Citation Snippet : {r['snippet']}\n" + "-" * 80)


if __name__ == "__main__":
    main()
