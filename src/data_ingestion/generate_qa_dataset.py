"""
IndiTrade AI - Phase 1: Automated Policy Q&A Dataset Generator
Generates high-quality Question & Answer pairs from real DGFT Trade Notifications
and PIB Press Releases (`dgft_policy_chunks.jsonl` & `pib_press_releases.jsonl`)
using the Groq API (`llama-3.3-70b-versatile` / `llama-3.1-8b-instant`).

Output format: JSON Lines (`data/processed/policy_qa_dataset.jsonl`)
"""

import os
import json
import time
import argparse
import logging
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PolicyQAGenerator")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = """You are an expert Indian International Trade, Customs, and DGFT Policy Compliance Advisor.
Your task is to generate 2 to 3 realistic, high-quality, professional Question & Answer (Q&A) pairs based STRICTLY on the provided government policy document chunk (DGFT Notification, Trade Notice, or PIB Press Release).

Rules:
1. Questions should represent what real Indian exporters, importers, customs clearing agents, or trade analysts would ask.
2. Answers must be detailed, factual, and directly supported by the text in the provided snippet. Include exact notification numbers, dates, HS codes, or policy mandates where mentioned.
3. If the snippet contains minimal substance or is boilerplate text, return an empty JSON list: []
4. Output MUST be strictly valid JSON array format without any markdown backticks or preamble.
Example format:
[
  {
    "question": "What is the mandatory operationalization date for the new DGFT Global Tariff module?",
    "answer": "As mandated by Trade Notice No. 01/2025-26, the operationalization of the new DGFT Global Tariff module takes effect immediately..."
  }
]
"""

def call_groq_api(chunk_text: str, doc_metadata: Dict[str, Any], api_key: str, model: str = "llama-3.3-70b-versatile", max_retries: int = 3) -> List[Dict[str, str]]:
    """Calls Groq API to generate Q&A pairs with rate limit & retry handling."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Truncate to maximum 3000 characters (~750 tokens) to guarantee staying well below Groq's TPM limits
    truncated_text = chunk_text[:3000] + ("..." if len(chunk_text) > 3000 else "")
    
    user_content = f"""Document Metadata:
- Source Type: {doc_metadata.get('doc_type', 'DGFT/PIB Document')}
- Document Reference: {doc_metadata.get('ref_id', 'N/A')}
- Title/Subject: {doc_metadata.get('title', 'N/A')}
- Date: {doc_metadata.get('date', 'N/A')}

Document Snippet:
{truncated_text}

Generate 2 to 3 factual Q&A pairs in exact JSON array format:"""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.3,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"} if "70b" in model or "8b" in model else None
    }
    # For models that require top-level json_object or json array, handle clean parsing
    if payload["response_format"] is not None:
        # If response_format json_object is requested, prompt explicitly to wrap in {"qa_pairs": [...]}
        payload["messages"][0]["content"] += '\nWrap your JSON array in an object like: {"qa_pairs": [{"question": "...", "answer": "..."}]}'

    for attempt in range(max_retries):
        try:
            resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=45)
            if resp.status_code == 429:
                wait_sec = (attempt + 1) * 5
                logger.warning(f"Groq API Rate Limit (429). Retrying in {wait_sec}s...")
                time.sleep(wait_sec)
                continue
                
            if resp.status_code != 200:
                # If 70b versatile fails or throws format error, fallback once to 8b
                if attempt == max_retries - 1 and model != "llama-3.1-8b-instant":
                    logger.warning(f"Groq API returned {resp.status_code}: {resp.text}. Falling back to llama-3.1-8b-instant...")
                    return call_groq_api(chunk_text, doc_metadata, api_key, model="llama-3.1-8b-instant", max_retries=2)
                logger.error(f"Groq API HTTP {resp.status_code}: {resp.text}")
                time.sleep(2)
                continue
                
            data = resp.json()
            raw_content = data["choices"][0]["message"]["content"].strip()
            
            # Clean possible markdown formatting
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:]
            if raw_content.startswith("```"):
                raw_content = raw_content[3:]
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3]
            raw_content = raw_content.strip()
            
            parsed = json.loads(raw_content)
            if isinstance(parsed, dict) and "qa_pairs" in parsed:
                return parsed["qa_pairs"]
            elif isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                # Look for first list value inside dict
                for k, v in parsed.items():
                    if isinstance(v, list):
                        return v
            return []
            
        except json.JSONDecodeError as je:
            logger.debug(f"JSON Parse Error on attempt {attempt+1}: {je}")
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error calling Groq API on attempt {attempt+1}: {e}")
            time.sleep(2)
            
    return []

def load_input_chunks(dgft_path: str, pib_path: str) -> List[Dict[str, Any]]:
    """Loads and standardizes chunks from both DGFT and PIB JSONL files."""
    chunks = []
    
    if os.path.exists(dgft_path):
        with open(dgft_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("clean_text", "").strip()
                if len(text) < 60:
                    continue
                chunks.append({
                    "doc_type": "DGFT_POLICY",
                    "ref_id": row.get("notification_no", row.get("chunk_id", "DGFT")),
                    "title": row.get("subject", "DGFT Trade Notification"),
                    "date": row.get("date", row.get("financial_year", "2024-25")),
                    "chunk_text": text,
                    "source_path": row.get("pdf_path", dgft_path)
                })
        logger.info(f"Loaded {len(chunks)} valid policy chunks from {dgft_path}")
    else:
        logger.warning(f"DGFT policy file not found: {dgft_path}")
        
    pib_count = 0
    if os.path.exists(pib_path):
        with open(pib_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("clean_text", "").strip()
                if len(text) < 60:
                    continue
                chunks.append({
                    "doc_type": "PIB_PRESS_RELEASE",
                    "ref_id": str(row.get("prid", "PIB")),
                    "title": row.get("title", "PIB Trade & Export Release"),
                    "date": row.get("date", "2024"),
                    "chunk_text": text,
                    "source_path": row.get("url", pib_path)
                })
                pib_count += 1
        logger.info(f"Loaded {pib_count} valid press releases from {pib_path}")
    else:
        logger.warning(f"PIB releases file not found: {pib_path}")
        
    return chunks

def main():
    parser = argparse.ArgumentParser(description="IndiTrade AI - Policy Q&A Dataset Generator (Groq API)")
    parser.add_argument("--dgft-path", type=str, default="data/processed/dgft_policy_chunks.jsonl", help="Path to DGFT chunks JSONL")
    parser.add_argument("--pib-path", type=str, default="data/processed/pib_press_releases.jsonl", help="Path to PIB releases JSONL")
    parser.add_argument("--output-path", type=str, default="data/processed/policy_qa_dataset.jsonl", help="Output path for generated Q&A JSONL")
    parser.add_argument("--model", type=str, default="llama-3.3-70b-versatile", help="Groq model to use for Q&A generation")
    parser.add_argument("--max-chunks", type=int, default=0, help="Maximum chunks to process (0 = all)")
    parser.add_argument("--sleep", type=float, default=1.2, help="Sleep seconds between Groq API calls to avoid rate limiting")
    args = parser.parse_args()
    
    api_key = os.getenv("GROQ_API_KEY2") or os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY1")
    if not api_key:
        logger.error("CRITICAL: Neither GROQ_API_KEY2 nor GROQ_API_KEY is set in .env! Please check environment configuration.")
        return
        
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    chunks = load_input_chunks(args.dgft_path, args.pib_path)
    if not chunks:
        logger.error("No valid input chunks found! Exiting.")
        return
        
    if args.max_chunks > 0:
        chunks = chunks[:args.max_chunks]
        logger.info(f"Limiting Q&A generation to first {args.max_chunks} chunks.")
        
    logger.info(f"Starting Q&A generation across {len(chunks)} chunks using Groq ({args.model})...")
    
    total_qa_generated = 0
    start_time = time.time()
    
    # Track existing processed IDs if resuming or appending
    existing_qa_count = 0
    processed_ref_ids = set()
    if os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        row = json.loads(line)
                        existing_qa_count += 1
                        if "source_ref_id" in row:
                            processed_ref_ids.add(str(row["source_ref_id"]))
                    except:
                        pass
        if existing_qa_count > 0:
            logger.info(f"Output file {args.output_path} already has {existing_qa_count} Q&A pairs across {len(processed_ref_ids)} unique documents. Resuming clean generation...")
            
    with open(args.output_path, "a", encoding="utf-8") as out_file:
        for idx, chunk in enumerate(chunks, 1):
            if str(chunk["ref_id"]) in processed_ref_ids:
                continue
                
            logger.info(f"[{idx}/{len(chunks)}] Processing {chunk['doc_type']} | Ref: {chunk['ref_id'][:30]}...")
            
            qa_pairs = call_groq_api(chunk["chunk_text"], chunk, api_key, model=args.model)
            if qa_pairs:
                processed_ref_ids.add(str(chunk["ref_id"]))
            
            if qa_pairs:
                for qa in qa_pairs:
                    if not isinstance(qa, dict) or not qa.get("question") or not qa.get("answer"):
                        continue
                        
                    entry = {
                        "qa_id": f"qa_{chunk['doc_type'].lower()}_{chunk['ref_id']}_{total_qa_generated+1}",
                        "doc_type": chunk["doc_type"],
                        "source_ref_id": chunk["ref_id"],
                        "source_title": chunk["title"],
                        "source_date": chunk["date"],
                        "question": qa["question"].strip(),
                        "answer": qa["answer"].strip(),
                        "context_snippet": chunk["chunk_text"][:500] + ("..." if len(chunk["chunk_text"]) > 500 else ""),
                        "generator_model": args.model,
                        "generated_at": datetime.now().isoformat()
                    }
                    out_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    out_file.flush()
                    total_qa_generated += 1
                    
            if idx < len(chunks):
                time.sleep(args.sleep)
                
    elapsed = time.time() - start_time
    logger.info("=== Q&A DATASET GENERATION COMPLETE ===")
    logger.info(f"Total Chunks Processed : {len(chunks)}")
    logger.info(f"Total Q&A Pairs Generated: {total_qa_generated}")
    logger.info(f"Total Q&A in Output File : {existing_qa_count + total_qa_generated}")
    logger.info(f"Time Taken               : {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    logger.info(f"Saved to                 : {args.output_path}")

if __name__ == "__main__":
    main()
