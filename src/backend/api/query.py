from fastapi import APIRouter
from pydantic import BaseModel
import logging
import os
import requests
from src.backend.database import qdrant

logger = logging.getLogger("api.query")
router = APIRouter()

class QueryRequest(BaseModel):
    question: str
    
@router.post("/")
async def query_policy(req: QueryRequest):
    # 1. RAG Context Retrieval
    context = ""
    try:
        if qdrant and hasattr(qdrant, 'search'):
            # Fetch context from Qdrant if available
            results = qdrant.search(
                collection_name="dgft_policy",
                query_vector=[0.0]*768, # Mock vector for demo since embeddings aren't fully integrated here
                limit=3
            )
            context = "\n".join([r.payload.get("text", "") for r in results])
    except Exception as e:
        logger.warning(f"RAG Retrieval failed or mocked: {e}")
        
    prompt = f"### Instruction:\nYou are an expert Indian Foreign Trade Policy assistant.\n\n### Context:\n{context}\n\n### Question:\n{req.question}\n\n### Answer:\n"
    
    # 2. Query Hugging Face API
    hf_token = os.getenv("HF_TOKEN")
    model_id = "Yash1bajpai/Inditrade-Llama-3.2-1B-Policy-Merged"
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    
    try:
        logger.info(f"Querying HF Serverless API: {model_id}")
        response = requests.post(
            api_url,
            headers={"Authorization": f"Bearer {hf_token}"},
            json={"inputs": prompt, "parameters": {"max_new_tokens": 300, "temperature": 0.2}},
            timeout=10
        )
        
        if response.status_code == 503:
            logger.warning("HF API 503 Cold Start Detected. Falling back to Groq API...")
            return fallback_query(req.question, context)
            
        response.raise_for_status()
        data = response.json()
        answer = data[0].get("generated_text", "").split("### Answer:\n")[-1]
        
        return {"answer": answer, "source": "Hugging Face (Fine-tuned)"}
        
    except requests.exceptions.Timeout:
        logger.warning("HF API Timeout. Falling back to Groq API...")
        return fallback_query(req.question, context)
    except Exception as e:
        logger.warning(f"HF API Failed ({e}). Falling back to Groq API...")
        return fallback_query(req.question, context)

def fallback_query(question, context):
    groq_api_key = os.getenv("GROQ_API_KEY1")
    if not groq_api_key or len(groq_api_key) < 10:
        return {"answer": "Error: Both Hugging Face and Groq Fallback APIs are unavailable.", "source": "Error"}
        
    try:
        import groq
        client = groq.Groq(api_key=groq_api_key)
        sys_prompt = "You are an expert Indian Trade Policy assistant."
        if context:
            sys_prompt += f" Use this context: {context}"
            
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": question}
            ],
            model="llama-3.1-8b-instant",
        )
        return {"answer": chat_completion.choices[0].message.content, "source": "Groq Fallback (Base Llama 3)"}
    except Exception as e:
        logger.error(f"Groq fallback failed: {e}")
        return {"answer": "Both primary and fallback APIs failed.", "source": "Error"}
