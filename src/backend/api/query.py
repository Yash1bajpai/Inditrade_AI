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

    context = ""
    citation_str = "Knowledge Base Error"
    try:
        if qdrant and hasattr(qdrant, 'search'):

            hf_token = os.getenv("HF_TOKEN")
            emb_response = requests.post(
                "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5/pipeline/feature-extraction",
                headers={"Authorization": f"Bearer {hf_token}"},
                json={"inputs": req.question},
                timeout=5
            )
            query_vector = emb_response.json() if emb_response.status_code == 200 else [0.0]*384

            if emb_response.status_code == 200:
                results = qdrant.search(
                    collection_name="trade_policy_compliance",
                    query_vector=query_vector,
                    limit=3
                )

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
                citation_str = " | ".join(set(citations)) if citations else "Unknown Source"
    except Exception as e:
        logger.warning(f"RAG Retrieval failed or mocked: {e}")
        citation_str = "Knowledge Base Error"

    prompt = f"### Instruction:\nYou are an expert Indian Foreign Trade Policy assistant.\n\n### Context:\n{context}\n\n### Question:\n{req.question}\n\n### Answer:\n"

    hf_token = os.getenv("HF_TOKEN")
    model_id = "Yash1bajpai/Inditrade-Llama-3.2-1B-Policy-Merged"
    api_url = f"https://router.huggingface.co/hf-inference/models/{model_id}"

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
            return fallback_query(req.question, context, citation_str)

        response.raise_for_status()
        data = response.json()
        answer = data[0].get("generated_text", "").split("### Answer:\n")[-1]

        return {"answer": answer, "source": "Hugging Face", "citation": citation_str}

    except requests.exceptions.Timeout:
        logger.warning("HF API Timeout. Falling back to Groq API...")
        return fallback_query(req.question, context, citation_str)
    except Exception as e:
        logger.warning(f"HF API Failed ({e}). Falling back to Groq API...")
        return fallback_query(req.question, context, citation_str)

def fallback_query(question, context, citation_str=""):
    groq_api_key = os.getenv("GROQ_API_KEY1")
    if not groq_api_key or len(groq_api_key) < 10:
        return {"answer": "Error: Both Hugging Face and Groq Fallback APIs are unavailable.", "source": "Error", "citation": ""}

    try:
        import groq
        client = groq.Groq(api_key=groq_api_key)
        sys_prompt = (
            "You are an expert Indian Trade Policy assistant. "
            "SECURITY MEASURE: You MUST strictly refuse to answer any questions that are not related to Indian Trade, DGFT, Import/Export policy, tariffs, or customs. "
            "If the user asks you to write code (like Python), write essays, translate unrelated text, or asks general knowledge questions, politely decline and state that you are only authorized to assist with trade policy. "
        )
        if context:
            sys_prompt += f" Use this context: {context}"

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": question}
            ],
            model="llama-3.1-8b-instant",
        )
        return {"answer": chat_completion.choices[0].message.content, "source": "Groq Fallback", "citation": citation_str}
    except Exception as e:
        logger.error(f"Groq fallback failed: {e}")
        return {"answer": "Both primary and fallback APIs failed.", "source": "Error"}

