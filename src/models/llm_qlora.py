"""
IndiTrade AI - Phase 4: Llama-3.2-1B QLoRA Fine-Tune (Policy Q&A)
Fine-tunes meta-llama/Llama-3.2-1B on data/processed/policy_qa_dataset.jsonl
using 4-bit QLoRA (bitsandbytes + PEFT). Auto-detects bf16 vs fp16 support
so the same script runs correctly on T4 (Colab/Lightning cheap tier, fp16)
or L4/A100 (bf16).

NOTE: Llama-3.2-1B is a GATED model on HuggingFace. Before running:
  1. Request access at https://huggingface.co/meta-llama/Llama-3.2-1B
  2. huggingface-cli login  (or set HF_TOKEN env var)

Dry run (verify pipeline works before spending real GPU time):
    python qlora_finetune.py --max-samples 100 --epochs 1

Full run:
    python qlora_finetune.py
"""

import os
import sys
import json
import argparse
from datetime import datetime

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_MODEL = "meta-llama/Llama-3.2-1B"
MAX_SEQ_LEN = 768

PROMPT_TEMPLATE = """### Instruction:
You are a Foreign Trade Policy expert. Answer the question about Indian trade
regulations using only the context below.

### Context:
{context}

### Question:
{question}

### Answer:
"""


def load_qa_dataset(path, max_samples=None):
    """Loads policy_qa_dataset.jsonl and returns train/val splits."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"[*] Loaded {len(records)} QA pairs from {path}")

    if max_samples:
        records = records[:max_samples]
        print(f"[*] DRY RUN: limited to first {max_samples} samples")

    # Genuine held-out split, not self-evaluation (same discipline as Module A)
    split_idx = int(len(records) * 0.95)
    train_records = records[:split_idx]
    val_records = records[split_idx:]
    print(f"[*] Train: {len(train_records)} | Val (held-out): {len(val_records)}")
    return train_records, val_records


def format_example(record, tokenizer):
    """Builds prompt+answer, tokenizes, and masks prompt tokens from the loss
    so the model is only penalized for generating the answer, not the prompt."""
    prompt = PROMPT_TEMPLATE.format(
        context=record.get("context_snippet", "")[:500],
        question=record["question"],
    )
    answer = record["answer"].strip() + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids

    input_ids = input_ids[:MAX_SEQ_LEN]
    labels = labels[:MAX_SEQ_LEN]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def main():
    parser = argparse.ArgumentParser(description="IndiTrade AI - Llama-3.2-1B QLoRA Fine-Tune")
    parser.add_argument("--data-path", type=str, default="data/processed/policy_qa_dataset.jsonl")
    parser.add_argument("--output-dir", type=str, default="models/llm_output/final_adapter")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit dataset size (dry run)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--push-to-hub", type=str, default=None,
                         help="HF repo id to push adapter to, e.g. yourname/inditrade-policy-llama32-1b")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=== [PHASE 4] LLAMA-3.2-1B QLoRA POLICY Q&A FINE-TUNE ===")

    if not torch.cuda.is_available():
        print("[FATAL] No CUDA GPU detected. QLoRA 4-bit requires a GPU. Stopping.")
        sys.exit(1)

    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"[*] GPU: {torch.cuda.get_device_name(0)} | bf16 supported: {use_bf16}")

    train_records, val_records = load_qa_dataset(args.data_path, args.max_samples)

    print(f"[*] Loading tokenizer + base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("[*] Tokenizing dataset (masking prompt tokens from loss)...")
    train_ds = Dataset.from_list([format_example(r, tokenizer) for r in train_records])
    val_ds = Dataset.from_list([format_example(r, tokenizer) for r in val_records])

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, label_pad_token_id=-100)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=use_bf16,
        fp16=not use_bf16,
        report_to="none",
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print(f"[*] Starting training: {len(train_ds)} train / {len(val_ds)} val | {args.epochs} epochs")
    train_result = trainer.train()

    print("[*] Running final evaluation on held-out val set...")
    eval_metrics = trainer.evaluate()
    eval_loss = eval_metrics.get("eval_loss")
    perplexity = float(torch.exp(torch.tensor(eval_loss))) if eval_loss is not None else None

    print("\n=== FINAL HELD-OUT VALIDATION METRICS ===")
    print(f"  * Eval Loss   : {eval_loss:.4f}")
    print(f"  * Perplexity  : {perplexity:.4f}")

    print(f"[*] Saving LoRA adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Proof-of-life: one real generation on a held-out question, per plan's proof requirement
    sample_output = None
    if len(val_records) > 0:
        sample_rec = val_records[0]
        prompt = PROMPT_TEMPLATE.format(
            context=sample_rec.get("context_snippet", "")[:500],
            question=sample_rec["question"],
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        gen_ids = model.generate(**inputs, max_new_tokens=150, do_sample=False)
        generated = tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        sample_output = {
            "question": sample_rec["question"],
            "ground_truth_answer": sample_rec["answer"],
            "model_generated_answer": generated.strip(),
        }
        print("\n=== SAMPLE GENERATION (held-out val question) ===")
        print(f"  Q: {sample_output['question']}")
        print(f"  Model A: {sample_output['model_generated_answer']}")

    hub_url = None
    if args.push_to_hub:
        print(f"[*] Pushing adapter to HF Hub: {args.push_to_hub}")
        try:
            model.push_to_hub(args.push_to_hub)
            tokenizer.push_to_hub(args.push_to_hub)
            hub_url = f"https://huggingface.co/{args.push_to_hub}"
            print(f"[*] Pushed successfully: {hub_url}")
        except Exception as e:
            print(f"[WARNING] HF Hub push failed: {e}")
            print("[WARNING] Adapter is still saved locally at", args.output_dir)

    meta = {
        "base_model": BASE_MODEL,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "n_train_samples": len(train_ds),
        "n_val_samples": len(val_ds),
        "epochs": args.epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.lr,
        "max_seq_len": MAX_SEQ_LEN,
        "used_bf16": use_bf16,
        "final_train_loss": train_result.metrics.get("train_loss"),
        "final_eval_loss": eval_loss,
        "final_perplexity": perplexity,
        "dry_run": args.max_samples is not None,
        "hf_hub_url": hub_url,
        "sample_generation": sample_output,
    }
    meta_path = os.path.join(args.output_dir, "training_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[Exported] training metadata -> {meta_path}")
    print("\n[COMPLETE] QLoRA fine-tune pipeline finished!")


if __name__ == "__main__":
    main()
