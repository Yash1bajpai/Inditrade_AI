import os
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("database")

load_dotenv()

class MockSupabase:
    def __init__(self):
        self.data = {}

    def table(self, table_name):
        return self

    def insert(self, data):
        return self

    def select(self, *args, **kwargs):
        return self

    def eq(self, *args, **kwargs):
        return self

    def execute(self):
        return {"data": [], "error": None}

import json

class MockQdrantRetriever:
    """Last-resort retriever: keyword overlap over the QA dataset.

    Only used when even the tracked BM25 artifact is missing (bare checkout).
    """

    supports_vector = False

    def __init__(self):
        self.documents = []
        try:
            if os.path.exists("data/processed/policy_qa_dataset.jsonl"):
                with open("data/processed/policy_qa_dataset.jsonl", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            self.documents.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to load local QA dataset: {e}")

    def search(self, collection_name, query_vector=None, limit=3, **kwargs):
        if not self.documents:
            return []
        
        query_text = kwargs.get("query_text", "")
        if not query_text:
            return []

        query_words = set(query_text.lower().split())
        scored_docs = []
        for doc in self.documents:
            text = (doc.get("question", "") + " " + doc.get("answer", "")).lower()
            score = sum(1 for w in query_words if w in text)
            if score > 0:
                scored_docs.append((score, doc))
        
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        class DummyHit:
            def __init__(self, doc, score_val, idx):
                self.id = idx
                self.score = float(score_val)
                self.payload = {
                    "text": f"Q: {doc.get('question', '')}\nA: {doc.get('answer', '')}",
                    "title": "Local QA Dataset",
                    "doc_type": "FAQ"
                }
        
        return [DummyHit(doc, score, idx) for idx, (score, doc) in enumerate(scored_docs[:limit])]

class Bm25LocalRetriever:
    """Offline fallback retriever: real BM25 over the chunked policy corpus.

    Backed by the lzma-compressed BM25 artifact tracked in git, so a fresh
    checkout (or the Render Docker image) can retrieve actual DGFT/PIB policy
    text with zero external services — no Qdrant cluster, no embedding API.
    """

    supports_vector = False

    def __init__(self):
        from src.rag.sparse_index import ProdBM25Index
        self._index = ProdBM25Index.load()
        logger.info(
            f"Bm25LocalRetriever ready: {len(self._index.docs)} indexed policy chunks"
        )

    def search(self, collection_name, query_vector=None, limit=3, **kwargs):
        query_text = kwargs.get("query_text", "")
        if not query_text:
            return []
        # Dedupe to one hit per parent document so the limited slots span
        # distinct notifications rather than neighboring chunks.
        hits = self._index.search(query_text, limit=limit * 3)
        seen_parents, deduped = set(), []
        for hit in hits:
            parent = hit.payload.get("parent_doc_id") or hit.payload.get("doc_id")
            if parent in seen_parents:
                continue
            seen_parents.add(parent)
            deduped.append(hit)
            if len(deduped) >= limit:
                break
        return deduped

def init_supabase():
    supabase_url = os.getenv("SUPABASE_URL", "")
    supabase_key = os.getenv("SUPABASE_KEY", "")

    if len(supabase_url) < 30 or "your_supabase" in supabase_url:
        logger.warning("Supabase connection failed (Fake/Placeholder Credentials Detected). Falling back to Mock Supabase.")
        return MockSupabase()

    try:
        from supabase import create_client, Client
        client: Client = create_client(supabase_url, supabase_key)
        return client
    except Exception as e:
        logger.warning(f"Supabase connection failed ({e}). Falling back to Mock Supabase.")
        return MockSupabase()

def _qdrant_url_blocked(qdrant_url: str) -> bool:
    """True when the configured URL is a placeholder or a known-dead cluster.

    Dead cluster IDs live in the comma-separated QDRANT_BLOCKED_URLS env var
    instead of being hardcoded here.
    """
    blocked_fragments = [
        fragment.strip()
        for fragment in os.getenv("QDRANT_BLOCKED_URLS", "").split(",")
        if fragment.strip()
    ]
    return (
        len(qdrant_url) < 30
        or "your_qdrant" in qdrant_url
        or any(fragment in qdrant_url for fragment in blocked_fragments)
    )

def init_qdrant():
    qdrant_url = os.getenv("QDRANT_URL", "")
    qdrant_key = os.getenv("QDRANT_API_KEY", "")

    if _qdrant_url_blocked(qdrant_url):
         logger.warning("Qdrant connection failed (Placeholder/Dead URL Detected). Using local BM25 retriever.")
         return _init_local_retriever()

    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

        client.get_collections()
        logger.info("Successfully connected to genuine Qdrant instance!")
        return client
    except Exception as e:
        logger.warning(f"Qdrant connection failed ({e}). Using local BM25 retriever.")
        return _init_local_retriever()

def _init_local_retriever():
    """Prefer the tracked BM25 index over the QA-dataset keyword mock."""
    try:
        return Bm25LocalRetriever()
    except FileNotFoundError as e:
        logger.warning(f"{e} Falling back to QA-dataset mock retriever.")
        return MockQdrantRetriever()
    except Exception as e:
        logger.warning(f"Failed to load BM25 artifact ({e}). Falling back to QA-dataset mock retriever.")
        return MockQdrantRetriever()

supabase = init_supabase()
qdrant = init_qdrant()

