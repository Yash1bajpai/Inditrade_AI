"""
IndiTrade AI - Phase 5: Method 1 Hugging Face Free Serverless Inference & Local Checkpoint Verifier
Tests your fine-tuned model either:
1. via Hugging Face Free Serverless Inference API (`api-inference.huggingface.co`) (Method 1 Production Test)
2. via Local LoRA Checkpoint (`models/checkpoints/inditrade-llama3-8b-policy-lora`) (Local Pre-push Verification)

Usage:
    # 1. Test against Hugging Face Free Serverless Inference API (Method 1)
    python src/models/test_hf_inference.py --mode cloud --model-id Yash1bajpai/Inditrade-Llama3-8B-Policy-Merged-16Bit --hf-token YOUR_TOKEN

    # 2. Test against local trained LoRA weights before pushing to hub
    python src/models/test_hf_inference.py --mode local --lora-path models/checkpoints/inditrade-llama3-8b-policy-lora
"""

import os
import sys
import json
import argparse
import urllib.request
import urllib.error

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

TEST_QUESTIONS = [
    "What is the export policy condition for Glufosinate Technical under DGFT Notification 14/2024-25?",
    "Under what conditions are imports of Yellow Peas permitted under HS Code 07131010?",
    "What is the procedure for allocation of quota for export of broken rice under Trade Notice 16/2023?"
]


def test_cloud_hf_inference(model_id: str, hf_token: str):
    """Tests Method 1: Hugging Face Free Serverless Inference API."""
    print("=" * 65)
    print(f"Testing Hugging Face Free Serverless Inference API")
    print(f"Target Model: https://api-inference.huggingface.co/models/{model_id}")
    print("=" * 65)

    if not hf_token:
        print("[WARNING] No --hf-token provided. If the repository is private or gated, requests may fail.")

    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {
        "Content-Type": "application/json",
    }
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    for idx, q in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[Q{idx}] {q}")
        prompt = f"### Instruction:\nYou are an expert Indian Foreign Trade Policy assistant for IndiTrade AI. Answer accurately.\n\n### Question:\n{q}\n\n### Answer:\n"
        
        payload = json.dumps({
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.2,
                "top_p": 0.9,
                "return_full_text": False
            }
        }).encode("utf-8")

        try:
            req = urllib.request.Request(api_url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=45) as response:
                result = json.loads(response.read().decode("utf-8"))
                if isinstance(result, list) and len(result) > 0:
                    ans_text = result[0].get("generated_text", "").strip()
                    print(f"[ANSWER] -> {ans_text}")
                else:
                    print(f"[RAW RESPONSE] -> {result}")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            print(f"[HTTP ERROR {e.code}] -> {err_body}")
            if e.code == 503:
                print("--> Note: Status 503 means Hugging Face is currently loading/warming up your model into memory. Try again in 30 seconds!")
        except Exception as e:
            print(f"[ERROR] -> {e}")


def test_local_lora_checkpoint(lora_path: str, base_model: str):
    """Tests local LoRA weights using Unsloth/PEFT before pushing to Hugging Face Hub."""
    print("=" * 65)
    print(f"Testing Local LoRA Checkpoint: {lora_path}")
    print("=" * 65)

    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA path not found: {lora_path}")

    import torch
    try:
        from unsloth import FastLanguageModel
        print("Loading local LoRA using Unsloth FastLanguageModel...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=lora_path,
            max_seq_length=1024,
            load_in_4bit=True
        )
        FastLanguageModel.for_inference(model)
    except ImportError:
        print("Unsloth not found, loading via Transformers + PEFT...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            load_in_4bit=True,
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, lora_path)
        model.eval()

    for idx, q in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[Q{idx}] {q}")
        prompt = f"### Instruction:\nYou are an expert Indian Foreign Trade Policy assistant for IndiTrade AI. Answer accurately.\n\n### Question:\n{q}\n\n### Answer:\n"
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.2, top_p=0.9, use_cache=True)
        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"[ANSWER] -> {gen_text}")


def parse_args():
    parser = argparse.ArgumentParser(description="Test IndiTrade AI Fine-Tuned Model (Method 1 Cloud vs Local)")
    parser.add_argument("--mode", type=str, choices=["cloud", "local"], default="cloud", help="Test Mode: cloud (HF API) or local (LoRA weights)")
    parser.add_argument("--model-id", type=str, default="Yash1bajpai/Inditrade-Llama3-8B-Policy-Merged-16Bit", help="HF Hub Repository ID")
    parser.add_argument("--hf-token", type=str, default=os.getenv("HF_TOKEN", ""), help="HF API Token")
    parser.add_argument("--lora-path", type=str, default="models/checkpoints/inditrade-llama3-8b-policy-lora", help="Local LoRA Checkpoint Path")
    parser.add_argument("--base-model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Base model for local loading")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "cloud":
        test_cloud_hf_inference(args.model_id, args.hf_token)
    else:
        test_local_lora_checkpoint(args.lora_path, args.base_model)
