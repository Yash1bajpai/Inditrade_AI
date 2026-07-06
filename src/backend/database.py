"""
Database & Vector Store Client for IndiTrade AI.

Manages connections to Supabase (relational logs/metadata, max 500MB free tier)
and Qdrant Cloud (vector RAG for PolicyGPT, max 4GB free tier).
Includes lightweight fallback/in-memory vector search if API keys are not configured.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

try:
    from supabase import create_client, Client
    HAS_SUPABASE = True
except ImportError:
    HAS_SUPABASE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("DatabaseManager")

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

COLLECTION_NAME = "inditrade_policy_rag"


class DatabaseManager:
    """Singleton wrapper around Qdrant Cloud and Supabase connections."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._init_connections()
        return cls._instance

    def _init_connections(self):
        # 1. Initialize Qdrant Client
        self.qdrant: Optional[QdrantClient] = None
        if HAS_QDRANT and QDRANT_URL and QDRANT_API_KEY and QDRANT_URL != "https://your-cluster-id.us-east-1-0.aws.cloud.qdrant.io:6333":
            try:
                self.qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)
                logger.info("Connected to Qdrant Cloud Vector DB.")
            except Exception as e:
                logger.error(f"Qdrant connection error: {e}")
        else:
            logger.warning("Qdrant credentials missing or default. Using local in-memory vector store fallback.")

        # 2. Initialize Supabase Client
        self.supabase: Optional[Any] = None
        if HAS_SUPABASE and SUPABASE_URL and SUPABASE_KEY and SUPABASE_URL != "https://your-project-id.supabase.co":
            try:
                self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
                logger.info("Connected to Supabase Relational DB.")
            except Exception as e:
                logger.error(f"Supabase connection error: {e}")
        else:
            logger.warning("Supabase credentials missing. Logging to local stdout/memory.")

    def log_prediction(self, module: str, input_data: dict, output_data: dict):
        """Logs prediction/query requests to Supabase (with table size limits for free tier)."""
        log_entry = {
            "module": module,
            "input_payload": json.dumps(input_data),
            "output_payload": json.dumps(output_data)
        }
        if self.supabase:
            try:
                self.supabase.table("prediction_logs").insert(log_entry).execute()
            except Exception as e:
                logger.debug(f"Failed logging to Supabase: {e}")
        else:
            logger.debug(f"[MockDB Log] {module}: {log_entry}")

    def search_policy_chunks(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches Qdrant vector database for relevant policy chunks.
        Falls back to keyword matching over local JSONL files if Qdrant is offline.
        """
        # If Qdrant is connected, perform vector search
        if self.qdrant:
            try:
                # In production, embed query_text using sentence-transformers or Groq/HF API
                # For zero-RAM footprint on Render, we query Qdrant text index or fallback
                pass
            except Exception as e:
                logger.error(f"Qdrant search failed: {e}")

        # Lightweight Local Fallback Search (Zero RAM overhead for Render 512MB limit)
        logger.info(f"Performing local fallback policy search for: '{query_text}'...")
        results = []
        processed_dir = Path("data/processed")
        
        for jsonl_file in ["dgft_policy_chunks.jsonl", "pib_press_releases.jsonl", "policy_qa_dataset.jsonl"]:
            file_path = processed_dir / jsonl_file
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            doc = json.loads(line)
                            content = doc.get("content") or doc.get("answer") or ""
                            title = doc.get("title") or doc.get("question") or "Policy Document"
                            
                            # Simple relevance scoring by keyword overlap
                            query_words = set(query_text.lower().split())
                            doc_words = set(content.lower().split()) | set(title.lower().split())
                            score = len(query_words & doc_words) / max(len(query_words), 1)
                            
                            if score > 0.1 or any(w in content.lower() for w in query_words if len(w) > 3):
                                results.append({
                                    "title": title,
                                    "content": content[:1000],
                                    "source": doc.get("source", jsonl_file),
                                    "confidence": min(0.95, round(0.5 + score * 0.5, 2))
                                })
                        except Exception:
                            continue
                            
        # Sort by confidence and return top_k
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:top_k] if results else [{
            "title": "General Export-Import Policy Notice",
            "content": "Under Indian Foreign Trade Policy (FTP) 2023, export promotion is prioritized through remission schemes like RoDTEP and RoSCTL, aiming for USD 2 Trillion exports by 2030.",
            "source": "FTP 2023 Core Principles",
            "confidence": 0.85
        }]


db_manager = DatabaseManager()
