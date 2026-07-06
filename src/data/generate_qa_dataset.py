"""
Synthetic Q&A Dataset Generator for IndiTrade AI (Llama-3.2-1B Fine-tuning).

Addresses the critical missing step: converting raw DGFT/PIB policy text chunks
into high-quality instruction-tuning (Q&A) pairs using Groq Free API (Llama 3.3 70B / Llama 3.1 8B).
Outputs to JSONL format compatible with HuggingFace datasets and SFTTrainer.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
    import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("SyntheticQAGenerator")

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Sample Indian Trade Policy chunks (Fallback/Seed data if DGFT PDFs aren't scraped yet)
SAMPLE_POLICY_CHUNKS = [
    """
    DGFT Notification No. 12/2015-20: In exercise of powers conferred by Section 3 of FT (D&R) Act, 1992, 
    the Central Government hereby makes the following amendments in the Foreign Trade Policy 2015-20. 
    Export of wheat and meslin under HS Code 1001 is prohibited with immediate effect to manage the overall 
    food security of the country and support the needs of neighbouring and other vulnerable countries. 
    However, export will be allowed on the basis of permission granted by the Government of India to other 
    countries to meet their food security needs and based on the request of their governments.
    """,
    """
    FEMA Regulations for FDI in E-commerce: Under the Foreign Exchange Management Act (FEMA), 100% FDI 
    is permitted under the automatic route in the marketplace model of e-commerce. However, FDI is not 
    permitted in the inventory-based model of e-commerce. A marketplace entity will be permitted to enter 
    into transactions with sellers registered on its platform on B2B basis. An e-commerce marketplace 
    entity will not exercise ownership or control over the inventory i.e. goods purported to be sold.
    """,
    """
    RBI Master Direction on Export of Goods and Services: All exporters must realize and repatriate the 
    full export value of goods or software to India within a period of 9 months from the date of export. 
    For Special Economic Zones (SEZs) and Export Oriented Units (EOUs), the realization period is also 
    9 months. In case of delay beyond 9 months, the exporter must seek extension from the authorized 
    dealer (AD) bank along with valid justification such as overseas buyer financial distress or legal disputes.
    """,
    """
    PIB Press Release - India-UAE CEPA: The India-UAE Comprehensive Economic Partnership Agreement (CEPA) 
    came into force on May 1, 2022. The agreement provides preferential market access for over 97% of its 
    tariff lines accounting for 99% of Indian exports to the UAE. Key labor-intensive sectors benefiting 
    include Gems and Jewellery, Textiles, Leather, Footwear, Sports Goods, Plastics, Furniture, and 
    Agricultural products. India imports crude oil, natural gas, and precious metals from UAE.
    """
]


class SyntheticQAGenerator:
    """Generates synthetic Q&A instruction pairs from text chunks using Groq API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key or GROQ_API_KEY
        self.model = model
        self.client = Groq(api_key=self.api_key) if (HAS_GROQ and self.api_key and self.api_key != "your_groq_free_api_key_here") else None
        
        if not self.client:
            logger.warning("Groq API client not initialized. Will use rule-based/mock generation if API key is missing.")

    def generate_qa_from_chunk(self, text_chunk: str, pairs_per_chunk: int = 3) -> List[Dict[str, str]]:
        """
        Sends a text chunk to Groq LLM asking it to generate `pairs_per_chunk` high-quality Q&A pairs.
        """
        text_chunk = text_chunk.strip()
        if not text_chunk:
            return []

        prompt = f"""You are an expert Indian Trade Law and Export-Import Policy analyst.
Read the following policy text chunk and generate exactly {pairs_per_chunk} high-quality, factual Question-and-Answer pairs for fine-tuning a legal/trade LLM.

Requirements:
1. Questions must be specific, realistic, and relevant to trade analysts or exporters.
2. Answers must be precise, professional, and strictly based on the provided text.
3. Output MUST be valid JSON as a list of objects with keys "question" and "answer", and NO OTHER TEXT OR MARKDOWN.

Text Chunk:
{text_chunk}

JSON Output:"""

        if self.client:
            try:
                response = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model,
                    temperature=0.3,
                    max_tokens=1024,
                    response_format={"type": "json_object"} if "70b" in self.model else None
                )
                content = response.choices[0].message.content.strip()
                
                # Parse JSON output
                if content.startswith("```json"):
                    content = content[7:-3].strip()
                elif content.startswith("```"):
                    content = content[3:-3].strip()
                    
                data = json.loads(content)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and "pairs" in data:
                    return data["pairs"]
                elif isinstance(data, dict) and "questions" in data:
                    return data["questions"]
                elif isinstance(data, dict):
                    # Try to find any list value in the dict
                    for val in data.values():
                        if isinstance(val, list):
                            return val
                return []
            except Exception as e:
                logger.error(f"Groq API generation failed: {e}")
                
        # Mock/Fallback generation if API key is missing or call fails
        logger.warning("WARNING: USING SYNTHETIC DATA — REAL SOURCE FAILED (Groq API call failed or key missing)")
        print("\n[WARNING: USING SYNTHETIC DATA — REAL SOURCE FAILED]\n")
        first_sentence = text_chunk.split(".")[0] + "."
        return [{
            "question": f"What does the policy state regarding: {first_sentence[:50]}...?",
            "answer": text_chunk
        }]

    def run_pipeline(self, chunks: Optional[List[str]] = None, output_path: str = "data/processed/policy_qa_dataset.jsonl") -> str:
        """
        Processes all chunks, generates Q&A pairs, applies rate limiting, and saves to JSONL.
        """
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        if chunks is None:
            # Try to load real extracted chunks from processed directory FIRST
            dgft_file = Path("data/processed/dgft_policy_chunks.jsonl")
            pib_file = Path("data/processed/pib_press_releases.jsonl")
            real_chunks = []
            if dgft_file.exists():
                with open(dgft_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            real_chunks.append(json.loads(line).get("content", ""))
            if pib_file.exists():
                with open(pib_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            real_chunks.append(json.loads(line).get("content", ""))
            
            real_chunks = [c for c in real_chunks if c]
            if real_chunks:
                logger.info(f"Loaded {len(real_chunks)} real text chunks from processed data.")
                chunks = real_chunks
            else:
                logger.warning("WARNING: USING SYNTHETIC DATA — REAL SOURCE FAILED (No processed policy/press release chunks found)")
                print("\n[WARNING: USING SYNTHETIC DATA — REAL SOURCE FAILED]\n")
                chunks = SAMPLE_POLICY_CHUNKS
                
        all_qa_pairs = []
        logger.info(f"Starting Synthetic Q&A Generation on {len(chunks)} text chunks...")
        
        for idx, chunk in enumerate(chunks, 1):
            logger.info(f"Processing chunk [{idx}/{len(chunks)}]...")
            pairs = self.generate_qa_from_chunk(chunk, pairs_per_chunk=3)
            
            if pairs:
                all_qa_pairs.extend(pairs)
                logger.info(f"Generated {len(pairs)} pairs from chunk {idx}.")
            
            # Groq free tier allows 30 req/min (2 seconds between calls is very safe)
            if self.client:
                time.sleep(2.0)
                
        # Write to JSONL
        with open(out_file, "w", encoding="utf-8") as f:
            for pair in all_qa_pairs:
                if "question" in pair and "answer" in pair:
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    
        logger.info(f"SUCCESS: Saved {len(all_qa_pairs)} instruction Q&A pairs to {out_file}")
        return str(out_file)


if __name__ == "__main__":
    generator = SyntheticQAGenerator()
    output_file = generator.run_pipeline()
    print(f"\n[+] Q&A Dataset generated: {output_file}")
