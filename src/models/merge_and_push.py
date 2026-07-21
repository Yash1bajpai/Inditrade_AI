import os
import sys
from dotenv import load_dotenv

# Ensure dependencies
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from huggingface_hub import HfApi
except ImportError:
    print("Installing required packages (torch, transformers, peft)...")
    os.system(f"{sys.executable} -m pip install -q torch transformers peft huggingface_hub")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from huggingface_hub import HfApi

load_dotenv()

# Use the finetune token if available
HF_TOKEN = os.getenv("HF_TOKEN_FINETUNE") or os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN not found in .env")
    sys.exit(1)

BASE_MODEL = "meta-llama/Llama-3.2-1B"
ADAPTER_DIR = "models/llm_output/final_adapter"
HUB_MODEL_ID = "Yash1bajpai/Inditrade-Llama-3.2-1B-Policy-Merged"

import traceback

print(f"Loading Base Model ({BASE_MODEL}) on CPU...")
try:
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu", # Keep on CPU for safety
        token=HF_TOKEN
    )
except Exception as e:
    print(f"Error loading base model: {e}")
    traceback.print_exc()
    sys.exit(1)

print(f"Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, token=HF_TOKEN)

print(f"Loading LoRA Adapter from {ADAPTER_DIR}...")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

print("Merging weights...")
merged_model = model.merge_and_unload()

print(f"Pushing merged model to HF Hub: {HUB_MODEL_ID} (PRIVATE)")
merged_model.push_to_hub(HUB_MODEL_ID, token=HF_TOKEN, private=True)
tokenizer.push_to_hub(HUB_MODEL_ID, token=HF_TOKEN, private=True)

print("✅ Merged model successfully pushed to Hugging Face Hub (Private Repository)!")
