from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
import logging
import os
import requests
import asyncio
import time
from src.backend.database import qdrant, supabase

logger = logging.getLogger("api.query")
router = APIRouter()

# Simple in-memory rate limiter
RATE_LIMIT = 10
RATE_LIMIT_WINDOW = 60
ip_requests = {}


def log_chat_to_supabase(session_id, user_message, bot_response):
    try:
        from src.backend.database import supabase
        if hasattr(supabase, 'table'):
            supabase.table('chat_history').insert({
                'session_id': session_id,
                'user_message': user_message,
                'bot_response': bot_response
            }).execute()
    except Exception as e:
        logger.error(f"Failed to log chat to Supabase: {e}")

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
        first_hop = forwarded.split(",")[0].strip() if forwarded else ""
        if first_hop:
            return first_hop
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
        if qdrant and hasattr(qdrant, 'search'):

            hf_token = os.getenv("HF_TOKEN")
            
            def fetch_embeddings():
                return requests.post(
                    "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5/pipeline/feature-extraction",
                    headers={"Authorization": f"Bearer {hf_token}"},
                    json={"inputs": req.question},
                    timeout=5
                )
            
            emb_response = await asyncio.to_thread(fetch_embeddings)
            query_vector = emb_response.json() if emb_response.status_code == 200 else [0.0]*384
            while isinstance(query_vector, list) and len(query_vector) > 0 and isinstance(query_vector[0], list):
                query_vector = query_vector[0]

            if emb_response.status_code == 200:
                def qdrant_search():
                    return qdrant.search(
                        collection_name="trade_policy_compliance",
                        query_vector=query_vector,
                        limit=3,
                        query_text=req.question
                    )
                
                results = await asyncio.to_thread(qdrant_search)

                contexts = []
                citations = []
                for r in results:
                    payload = r.payload
                    text = payload.get("text", "")
                    title = payload.get("title", "")
                    notif = payload.get("notification_no", "")
                    doc_type = payload.get("doc_type", "Policy")

                    contexts.append(text)
                    if title:
                        cite = f"{doc_type}: {title}"
                        if notif:
                            cite += f" (No. {notif})"
                        citations.append(cite)

                context = "\n".join(contexts)
                citation_str = " | ".join(set(citations)) if citations else ""
                grounded = bool(context.strip())
    except Exception as e:
        logger.warning(f"RAG Retrieval failed or mocked: {e}")
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
        logger.info(f"Querying HF Router API: {model_id}")

        def fetch_hf():
            messages = [{"role": "system", "content": (
                "You are an expert Indian Foreign Trade Policy assistant. "
                "Answer only questions related to Indian trade, DGFT, import/export policy, tariffs, or customs. "
                "Provide your answer in concise bullet points. Be extremely clear, short, and well-structured."
            )}]
            if context:
                messages.append({"role": "system", "content": f"Use this retrieved policy context:\n{context}"})
            messages.append({"role": "user", "content": req.question})
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

        res = {"answer": answer, "source": "Hugging Face", "citation": citation_str, "grounded": grounded}
        log_chat_to_supabase(client_ip, req.question, res["answer"])
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
            sys_prompt = (
                "You are an expert Indian Trade Policy assistant. "
                "SECURITY MEASURE: You MUST strictly refuse to answer any questions that are not related to Indian Trade, DGFT, Import/Export policy, tariffs, or customs. "
                "If the user asks you to write code (like Python), write essays, translate unrelated text, or asks general knowledge questions, politely decline and state that you are only authorized to assist with trade policy. "
                "CRITICAL FORMATTING INSTRUCTION: Always provide your answer in concise bullet points. Be extremely brief, clear, and well-structured."
            )
            if context:
                sys_prompt += f" Use this context: {context}"

            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": question}
                ],
                model="llama-3.1-8b-instant",
                timeout=30.0,
            )
            return chat_completion.choices[0].message.content
            
        answer = await asyncio.to_thread(run_groq)        
        res = {
            "answer": answer,
            "source": "Groq",
            "citation": citation_str,
            "grounded": grounded
        }
        log_chat_to_supabase(client_ip, question, res["answer"])
        return res

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=502, detail="Upstream language model request failed.")
