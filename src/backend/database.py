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

import json

class MockQdrantRetriever:
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
            def __init__(self, doc):
                self.payload = {
                    "text": f"Q: {doc.get('question', '')}\nA: {doc.get('answer', '')}",
                    "title": "Local QA Dataset",
                    "doc_type": "FAQ"
                }
        
        return [DummyHit(doc) for score, doc in scored_docs[:limit]]

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

def init_qdrant():
    qdrant_url = os.getenv("QDRANT_URL", "")
    qdrant_key = os.getenv("QDRANT_API_KEY", "")

    if len(qdrant_url) < 30 or "your_qdrant" in qdrant_url or "iocdpqfxoavhjcvhwuevlp" in qdrant_url:
         logger.warning("Qdrant connection failed (Placeholder/Dead URL Detected). Falling back to Mock Retriever.")
         return MockQdrantRetriever()

    try:
        from qdrant_client import QdrantClient

        if len(qdrant_url) > 30 and "your_qdrant" not in qdrant_url and "iocdpqfxoavhjcvhwuevlp" not in qdrant_url:
            client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
        else:
            client = QdrantClient(path="data/cache/qdrant_index")

        client.get_collections()
        logger.info("Successfully connected to genuine Qdrant instance!")
        return client
    except Exception as e:
        logger.warning(f"Qdrant connection failed ({e}). Falling back to Mock Retriever.")
        return MockQdrantRetriever()

supabase = init_supabase()
qdrant = init_qdrant()

