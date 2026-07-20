import os
import logging
from dotenv import load_dotenv

# Set up loud logging as strictly requested
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("database")

load_dotenv()

# --- SUPABASE MOCK FALLBACK ---
class MockSupabase:
    def __init__(self):
        self.data = {}
        
    def table(self, table_name):
        return self

# --- QDRANT MOCK FALLBACK ---
class MockQdrantRetriever:
    def search(self, query, top_k=3, **kwargs):
        logger.warning(f"MockRetriever called for query. Returning empty context.")
        return []

def init_supabase():
    supabase_url = os.getenv("SUPABASE_URL", "")
    supabase_key = os.getenv("SUPABASE_KEY", "")
    
    # Check if the keys look like genuine JWT/URLs or placeholders
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
        # 3 second timeout so we don't hang the server boot
        client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=3.0)
        # Verify connection by attempting to list collections
        client.get_collections()
        logger.info("Successfully connected to genuine Qdrant instance!")
        return client
    except Exception as e:
        logger.warning(f"Qdrant connection failed ({e}). Falling back to Mock Retriever.")
        return MockQdrantRetriever()

# Initialize globals (lazy-loading evaluated on boot for fast endpoint response)
supabase = init_supabase()
qdrant = init_qdrant()
