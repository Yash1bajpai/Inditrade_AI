from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
import logging
import os
import requests
import asyncio
import time
from src.backend.database import qdrant, supabase
from src.rag.sparse_index import ProdBM25Index
from src.rag.grounding import evidence_messages, verified_answer

logger = logging.getLogger("api.query")
router = APIRouter()

# Hybrid retrieval budget: fetch dense + sparse candidates, fuse with RRF,
# keep the top FINAL_CONTEXTS parents as generation context.
DENSE_HITS = 8
SPARSE_HITS = 8
FINAL_CONTEXTS = 5
RRF_K = 60

_sparse_index = None
_sparse_loaded = False


def get_sparse_index():
    """Lazily load the tracked BM25 artifact once; None when unavailable."""
    global _sparse_index, _sparse_loaded
    if not _sparse_loaded:
        _sparse_loaded = True
        try:
            _sparse_index = ProdBM25Index.load()
            logger.info(f"ProdBM25Index loaded: {len(_sparse_index.docs)} policy chunks")
        except Exception as e:
            logger.warning(f"BM25 artifact unavailable ({e}); sparse retrieval disabled.")
            _sparse_index = None
    return _sparse_index


def _validated_query_vector(value):
    """Accept one BGE sentence vector, never token matrices or zero vectors."""
    import math

    # Providers can wrap a single sentence vector in singleton batch axes.
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or len(value) != 384:
        raise ValueError("Expected one 384-dimensional BGE sentence vector")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in value):
        raise ValueError("Embedding must contain finite numbers")
    norm = math.sqrt(sum(v * v for v in value))
    if norm == 0:
        raise ValueError("Embedding must not be a zero vector")
    return [v / norm for v in value]


def _rrf_fuse(dense_hits, sparse_hits, rrf_k: int = RRF_K):
    """Reciprocal Rank Fusion over dense and sparse hit lists.

    Hits are objects exposing .payload. Fusion keys on the parent document so
    multiple chunks of the same notification cannot crowd out other sources.
    Returns payloads ordered by fused score.
    """
    scores, payloads = {}, {}
    for hits in (dense_hits, sparse_hits):
        for rank, hit in enumerate(hits, start=1):
            payload = hit.payload or {}
            key = payload.get("parent_doc_id") or payload.get("doc_id") or f"hit_{id(payload)}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            if key not in payloads:
                payloads[key] = payload
    return [payloads[k] for k in sorted(scores, key=scores.get, reverse=True)]

# Simple in-memory rate limiter
RATE_LIMIT = 10
RATE_LIMIT_WINDOW = 60
ip_requests = {}


def log_chat_to_supabase(session_id, user_message, bot_response):
    try:
        from src.backend.database import supabase
        if supabase is not None and hasattr(supabase, 'table'):
            supabase.table('chat_history').insert({
                'session_id': session_id,
                'user_message': user_message,
                'bot_response': bot_response
            }).execute()
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to log chat to Supabase: {type(e).__name__}")
        return False

class QueryRequest(BaseModel):
    question: str = Field(..., max_length=500, description="The query string (max 500 chars).")

def _resolve_client_ip(request: Request) -> str:
    """Client IP for rate limiting.

    Direct connections use request.client.host. Behind a reverse proxy that
    overwrites X-Forwarded-For (Render does), set TRUST_PROXY=true so the
    limiter keys on the real client instead of the proxy IP — otherwise every
    visitor shares one bucket.
    """
    if os.getenv("TRUST_PROXY", "").lower() in ("1", "true", "yes"):
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            # Use the LAST hop: it is appended by the closest trusted proxy
            # (e.g. Render), whereas earlier entries are client-supplied and
            # trivially spoofable.
            return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"

def _enforce_access_key(request: Request):
    """Optional shared-secret gate for the LLM endpoint.

    Set API_ACCESS_KEY in the environment to require clients to send a
    matching X-API-Key header. When the variable is unset the endpoint stays
    open, preserving current demo behaviour.
    """
    expected = os.getenv("API_ACCESS_KEY", "")
    if not expected:
        return
    provided = request.headers.get("x-api-key", "")
    if provided != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")

@router.post("/")
async def query_policy(req: QueryRequest, request: Request):
    _enforce_access_key(request)
    client_ip = _resolve_client_ip(request)
    current_time = time.time()

    # Rate Limiting Logic
    # Evict stale IP entries to prevent unbounded memory growth
    STALE_THRESHOLD = RATE_LIMIT_WINDOW * 5  # 5 minutes
    stale_ips = [ip for ip, ts_list in ip_requests.items()
                 if not ts_list or current_time - ts_list[-1] > STALE_THRESHOLD]
    for ip in stale_ips:
        del ip_requests[ip]

    if client_ip not in ip_requests:
        ip_requests[client_ip] = []
    ip_requests[client_ip] = [t for t in ip_requests[client_ip] if current_time - t < RATE_LIMIT_WINDOW]

    if len(ip_requests[client_ip]) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")

    ip_requests[client_ip].append(current_time)

    context = ""
    citation_str = "Knowledge Base Error"
    # Tracks whether the answer is actually backed by retrieved policy documents.
    # Without this, an ungrounded LLM answer is indistinguishable from a grounded
    # one in the response body.
    grounded = False
    try:
        dense_hits = []
        if qdrant is not None and getattr(qdrant, "supports_vector", True) and hasattr(qdrant, "query_points"):
            # Dense retrieval is optional. Its API/embedding failures must not
            # prevent the independent local sparse search below from running.
            try:
                mode = os.getenv("QDRANT_EMBEDDING_MODE", "hf_bge")
                if mode == "cloud_e5":
                    from qdrant_client.models import Document
                    def cloud_search():
                        return qdrant.query_points(
                            collection_name="trade_policy_e5_small_v1",
                            query=Document(text=req.question, model="intfloat/multilingual-e5-small"),
                            limit=DENSE_HITS, with_payload=True,
                        ).points
                    dense_hits = await asyncio.to_thread(cloud_search)
                elif mode == "hf_bge":
                    hf_token = os.getenv("HF_TOKEN")

                    def fetch_embeddings():
                        return requests.post(
                            "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5/pipeline/feature-extraction",
                            headers={"Authorization": f"Bearer {hf_token}"},
                            json={"inputs": req.question},
                            timeout=5
                        )

                    emb_response = await asyncio.to_thread(fetch_embeddings)
                    if emb_response.status_code != 200:
                        logger.warning(
                            "HF embedding API returned %s; using sparse only.",
                            emb_response.status_code,
                        )
                    else:
                        query_vector = _validated_query_vector(emb_response.json())

                        def qdrant_search():
                            return qdrant.query_points(
                                collection_name="trade_policy_compliance",
                                query=query_vector,
                                limit=DENSE_HITS,
                                with_payload=True,
                            ).points

                        dense_hits = await asyncio.to_thread(qdrant_search)
                else:
                    raise ValueError("Unsupported QDRANT_EMBEDDING_MODE")
            except Exception as e:
                logger.warning("Dense retrieval failed (%s); using sparse only.", e)

        # Reuse the fallback retriever's in-memory index when present (no
        # second 20MB load on the 512MB Render container); otherwise load the
        # tracked artifact once.
        sparse = getattr(qdrant, "_index", None) or get_sparse_index()
        sparse_hits = []
        if sparse is not None:
            def sparse_search():
                return sparse.search(req.question, limit=SPARSE_HITS)
            sparse_hits = await asyncio.to_thread(sparse_search)

        fused = _rrf_fuse(dense_hits, sparse_hits)[:FINAL_CONTEXTS]
        contexts, citations, seen_parents = [], [], set()
        for payload in fused:
            text = payload.get("text", "")
            if not text:
                continue
            contexts.append(text)
            parent = payload.get("parent_doc_id") or payload.get("doc_id")
            title = payload.get("title", "")
            if title and parent not in seen_parents:
                seen_parents.add(parent)
                cite = f"{payload.get('doc_type', 'Policy')}: {title}"
                if payload.get("notification_no"):
                    cite += f" (No. {payload['notification_no']})"
                citations.append(cite)

        context = "\n".join(contexts)
        citation_str = " | ".join(citations) if citations else ""
        grounded = bool(context.strip())
    except Exception as e:
        logger.warning(f"RAG Retrieval failed: {e}")
        citation_str = ""
        grounded = False

    hf_token = os.getenv("HF_TOKEN")
    # The legacy per-model serverless route (router.huggingface.co/hf-inference/
    # models/<custom-model>) now returns 400 "Model not supported by provider
    # hf-inference" for causal LMs. Use the OpenAI-compatible router endpoint
    # with a publicly served instruct model instead.
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    api_url = "https://router.huggingface.co/v1/chat/completions"

    try:
        if not hf_token:
            raise ValueError("HF token not configured")
        logger.info(f"Querying HF Router API: {model_id}")

        def fetch_hf():
            messages = evidence_messages(req.question, context)
            return requests.post(
                api_url,
                headers={"Authorization": f"Bearer {hf_token}"},
                json={"model": model_id, "messages": messages,
                      "max_tokens": 300, "temperature": 0.2},
                timeout=15
            )

        response = await asyncio.to_thread(fetch_hf)

        if response.status_code == 503:
            logger.warning("HF API 503 Cold Start Detected. Falling back to Groq API...")
            res = await fallback_query(req.question, context, citation_str, client_ip, grounded)
            return res

        response.raise_for_status()
        data = response.json()
        answer = data["choices"][0]["message"]["content"].strip()
        answer, grounded = verified_answer(answer, context, req.question)
        res = {"answer": answer, "source": "Hugging Face", "citation": citation_str, "grounded": grounded}
        res["history_saved"] = bool(await asyncio.to_thread(log_chat_to_supabase, client_ip, req.question, res["answer"]))
        return res

    except requests.exceptions.Timeout:
        logger.warning("HF API Timeout. Falling back to Groq API...")
        res = await fallback_query(req.question, context, citation_str, client_ip, grounded)
        return res
    except Exception as e:
        logger.warning(f"HF API Failed ({e}). Falling back to Groq API...")
        res = await fallback_query(req.question, context, citation_str, client_ip, grounded)
        return res

async def fallback_query(question, context, citation_str="", client_ip="anonymous", grounded=False):
    groq_api_key = os.getenv("GROQ_API_KEY", os.getenv("GROQ_API_KEY1"))
    if not groq_api_key or len(groq_api_key) < 10:
        # Both upstreams are unusable. This is a server-side dependency failure,
        # so signal it with 503 rather than a 200 carrying an error string that
        # clients would have to parse out of the answer field.
        msg = "Both Hugging Face and Groq Fallback APIs are unavailable."
        logger.error(msg)
        log_chat_to_supabase(client_ip, question, f"Error: {msg}")
        raise HTTPException(status_code=503, detail=msg)

    try:
        import groq

        def run_groq():
            client = groq.Groq(api_key=groq_api_key)
            chat_completion = client.chat.completions.create(
                messages=evidence_messages(question, context),
                model="openai/gpt-oss-20b",
                max_completion_tokens=4096,
                reasoning_effort="low",
                timeout=30.0,
            )
            answer = (chat_completion.choices[0].message.content or "").strip()
            if not answer or chat_completion.choices[0].finish_reason == "length":
                raise ValueError("Language model returned an empty or truncated answer.")
            return answer

        answer = await asyncio.to_thread(run_groq)
        answer, grounded = verified_answer(answer, context, question)
        res = {
            "answer": answer,
            "source": "Groq",
            "citation": citation_str,
            "grounded": grounded
        }
        res["history_saved"] = bool(await asyncio.to_thread(log_chat_to_supabase, client_ip, question, res["answer"]))
        return res

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=502, detail="Upstream language model request failed.")
