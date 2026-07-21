"""
IndiTrade AI - Phase 5: Production QLoRA / Unsloth Fine-Tuning & Hugging Face Hub Exporter
Fine-tunes Llama-3-8B-Instruct (or Llama-3.1-8B-Instruct) on `data/processed/policy_qa_dataset.jsonl` (1,360 pairs)
using 4-bit QLoRA and Unsloth (2x-5x faster, 70% less VRAM).

Key Capabilities for Method 1 (100% Free Serverless Inference API):
1. Trains LoRA adapter on clean DGFT, PIB, and OCR policy Q&A dataset.
2. Saves LoRA checkpoint locally (`models/checkpoints/inditrade-llama3-8b-policy-lora`).
3. If `--push-to-hub` is provided, automatically pushes the LoRA adapter (and optionally merged 16-bit/4-bit model)
   to Hugging Face Hub (e.g. `Yash1bajpai/Inditrade-Llama3-8B-Policy`).
4. Once pushed, Hugging Face automatically hosts a Free Serverless Inference API endpoint (`api-inference.huggingface.co`)
   ready to be queried by our Vercel Next.js web application at zero cost!

Usage on Lightning AI Studio or Google Colab:
    # 1. Install dependencies
    pip install -r src/models/requirements_finetune.txt

    # 2. Dry-run test (verify dataset loading & model initialization with 50 steps)
    python src/models/fine_tune_unsloth.py --max-steps 50 --batch-size 2

    # 3. Full training + Push to Hugging Face Hub for Free Inference API
    python src/models/fine_tune_unsloth.py --epochs 3 --push-to-hub --hub-model-id Yash1bajpai/Inditrade-Llama3-8B-Policy --hf-token YOUR_HF_TOKEN
"""

import os
import sys
import json
import argparse
from datetime import datetime

import torch
from datasets import Dataset

UNSLOTH_AVAILABLE = False
try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template
    UNSLOTH_AVAILABLE = True
except ImportError:
    pass

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DEFAULT_BASE_MODEL = "unsloth/llama-3-8b-Instruct-bnb-4bit" if UNSLOTH_AVAILABLE else "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_DATASET_PATH = "data/processed/policy_qa_dataset.jsonl"
DEFAULT_OUTPUT_DIR = "models/checkpoints/inditrade-llama3-8b-policy-lora"

PROMPT_TEMPLATE = """### Instruction:
You are an expert Indian Foreign Trade Policy and DGFT regulatory assistant for IndiTrade AI. Answer the following policy question strictly based on the provided trade context, citing relevant HS codes, notification numbers, and regulatory conditions accurately.

### Context:
{context}

### Question:
{question}

### Answer:
{answer}"""

def load_qa_dataset(dataset_path: str):
    """Loads and verifies the 1,360-pair policy Q&A dataset."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Ensure you are running from project root.")

    rows = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                q = record.get("question", "").strip()
                ctx = record.get("context_snippet", "").strip()
                ans = record.get("answer", "").strip()
                if q and ans:
                    rows.append({"question": q, "context": ctx, "answer": ans})
            except json.JSONDecodeError:
                continue

    print(f"[OK] Successfully loaded {len(rows)} verified Q&A pairs from {dataset_path}")
    return Dataset.from_list(rows)

def format_dataset_for_training(dataset: Dataset, tokenizer):
    """Formats raw (question, context, answer) rows into prompt text."""
    def _apply_template(examples):
        texts = []
        for q, c, a in zip(examples["question"], examples["context"], examples["answer"]):
            formatted = PROMPT_TEMPLATE.format(context=c, question=q, answer=a)
            if hasattr(tokenizer, "eos_token") and tokenizer.eos_token:
                formatted += tokenizer.eos_token
            texts.append(formatted)
        return {"text": texts}

    return dataset.map(_apply_template, batched=True, desc="Formatting Q&A pairs into instruction prompt strings")

def train_with_unsloth(args):
    """Executes high-speed training using Unsloth (Recommended for Lightning AI / Colab)."""
    print(f"=== Initializing Unsloth FastLanguageModel ({args.model_name}) ===")
    max_seq_len = args.max_seq_len
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=max_seq_len,
        load_in_4bit=True,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=max_seq_len,
    )

    dataset = load_qa_dataset(args.dataset_path)
    formatted_dataset = format_dataset_for_training(dataset, tokenizer)

    from trl import SFTTrainer
    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        num_train_epochs=args.epochs if args.max_steps <= 0 else 1,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        save_strategy="epoch" if args.max_steps <= 0 else "steps",
        save_steps=100 if args.max_steps > 0 else 500,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        dataset_num_proc=2,
        args=training_args,
    )

    print("\n=== Starting QLoRA Fine-Tuning ===")
    trainer_stats = trainer.train()
    print(f"\n[OK] Training completed in {trainer_stats.metrics['train_runtime']:.2f} seconds.")

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[OK] Saved local LoRA adapter weights to: {args.output_dir}")

    if args.push_to_hub and args.hub_model_id:
        print(f"\n=== Pushing Fine-Tuned Model to Hugging Face Hub ({args.hub_model_id}) ===")
        if args.hf_token:
            from huggingface_hub import login
            login(token=args.hf_token)

        model.push_to_hub_merged(
            args.hub_model_id,
            tokenizer,
            save_method="lora",
            token=args.hf_token
        )
        print(f"[SUCCESS] Pushed LoRA adapter to https://huggingface.co/{args.hub_model_id}")

        if args.push_merged_16bit:
            merged_id = f"{args.hub_model_id}-Merged-16Bit"
            print(f"Merging LoRA weights and pushing 16-bit standalone model to {merged_id}...")
            model.push_to_hub_merged(
                merged_id,
                tokenizer,
                save_method="merged_16bit",
                token=args.hf_token
            )
            print(f"[SUCCESS] Pushed 16-bit merged model to https://huggingface.co/{merged_id}")
            print(f"--> Your Free Serverless Inference API endpoint is now: https://api-inference.huggingface.co/models/{merged_id}")

def train_with_standard_transformers(args):
    """Fallback training logic using standard Transformers + PEFT if Unsloth is not installed."""
    print(f"=== Initializing Standard Transformers QLoRA ({args.model_name}) ===")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    dataset = load_qa_dataset(args.dataset_path)
    formatted_dataset = format_dataset_for_training(dataset, tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        num_train_epochs=args.epochs if args.max_steps <= 0 else 1,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        save_strategy="epoch" if args.max_steps <= 0 else "steps",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        args=training_args,
    )

    print("\n=== Starting Standard QLoRA Fine-Tuning ===")
    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[OK] Saved local LoRA adapter to: {args.output_dir}")

    if args.push_to_hub and args.hub_model_id:
        print(f"\n=== Pushing LoRA Adapter to Hugging Face Hub ({args.hub_model_id}) ===")
        if args.hf_token:
            from huggingface_hub import login
            login(token=args.hf_token)
        model.push_to_hub(args.hub_model_id, token=args.hf_token)
        tokenizer.push_to_hub(args.hub_model_id, token=args.hf_token)
        print(f"[SUCCESS] Pushed LoRA adapter to https://huggingface.co/{args.hub_model_id}")

def parse_args():
    parser = argparse.ArgumentParser(description="IndiTrade AI Phase 5 Fine-Tuning Suite")
    parser.add_argument("--model-name", type=str, default=DEFAULT_BASE_MODEL, help="Base HuggingFace/Unsloth model name")
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH, help="Path to policy_qa_dataset.jsonl")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save LoRA weights")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=2, help="Per device train batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max-steps", type=int, default=-1, help="If > 0, override epochs and train for N steps (for dry run)")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--push-to-hub", action="store_true", help="Push model/adapter to Hugging Face Hub after training")
    parser.add_argument("--hub-model-id", type=str, default="Yash1bajpai/Inditrade-Llama3-8B-Policy", help="HF Repo ID")
    parser.add_argument("--hf-token", type=str, default=os.getenv("HF_TOKEN", ""), help="Hugging Face API Token")
    parser.add_argument("--push-merged-16bit", action="store_true", help="Merge LoRA and push 16-bit standalone model to HF Hub")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print("=" * 65)
    print("      INDITRADE AI - PHASE 5 QLORA FINE-TUNING SUITE")
    print("=" * 65)
    print(f"Base Model : {args.model_name}")
    print(f"Dataset    : {args.dataset_path}")
    print(f"Unsloth?   : {'YES (Enabled - 2x-5x Faster)' if UNSLOTH_AVAILABLE else 'NO (Fallback to Standard Transformers)'}")
    print("=" * 65)

    if UNSLOTH_AVAILABLE:
        train_with_unsloth(args)
    else:
        train_with_standard_transformers(args)

