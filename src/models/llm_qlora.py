"""
Llama-3.2-1B QLoRA Fine-tuning Module for PolicyGPT (Module B).

Fine-tunes Meta Llama-3.2-1B on the synthetic Indian trade policy Q&A dataset (`policy_qa_dataset.jsonl`)
using 4-bit QLoRA (PEFT + bitsandbytes + SFTTrainer).
Pushes trained adapter weights directly to HuggingFace Hub (`yashbajpai/inditrade-llama-3.2-1b`)
under the user's personal identity.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import load_dataset
    from trl import SFTTrainer
    HAS_LLM_LIBS = True
except ImportError:
    HAS_LLM_LIBS = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("QLoRATrainer")

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", "")

DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B"
DEFAULT_HUB_MODEL_ID = "yashbajpai/inditrade-llama-3.2-1b"


class TradeLLMTrainer:
    """Fine-tunes 1B parameter LLM via QLoRA on Indian trade policy corpus."""

    def __init__(self, 
                 dataset_path: str = "data/processed/policy_qa_dataset.jsonl",
                 output_dir: str = "models/llm_output",
                 hub_model_id: str = DEFAULT_HUB_MODEL_ID):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hub_model_id = hub_model_id

    def format_prompt(self, example: dict) -> str:
        """Formats QA pair into Llama-3 instruction format."""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are PolicyGPT, an expert AI assistant on Indian foreign trade policy, DGFT regulations, FEMA rules, and macroeconomic trends.<|eot_id|><|start_header_id|>user<|end_header_id|>
{example['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{example['answer']}<|eot_id|>"""

    def train_and_push(self, num_train_epochs: int = 3, push_to_hub: bool = True) -> str:
        """Executes QLoRA fine-tuning and pushes to HuggingFace Hub."""
        if not HAS_LLM_LIBS:
            logger.error("Transformers, PEFT, TRL, or PyTorch missing. Cannot run LLM fine-tuning.")
            raise ImportError("Missing required deep learning libraries.")
            
        if not self.dataset_path.exists():
            logger.error(f"Dataset missing at {self.dataset_path}. Run generate_qa_dataset.py first.")
            raise FileNotFoundError(f"Missing {self.dataset_path}")
            
        logger.info(f"Loading QA dataset from {self.dataset_path}...")
        dataset = load_dataset("json", data_files=str(self.dataset_path), split="train")
        
        logger.info(f"Initializing 4-bit Quantization Config for {DEFAULT_MODEL_ID}...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        try:
            logger.info(f"Attempting to load tokenizer for {DEFAULT_MODEL_ID}...")
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_ID, token=HF_TOKEN or None)
            model_id_to_use = DEFAULT_MODEL_ID
        except Exception as e:
            logger.warning(f"Could not load {DEFAULT_MODEL_ID} (gated/auth error: {e}). Falling back to open ungated TinyLlama/TinyLlama-1.1B-Chat-v1.0...")
            model_id_to_use = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            tokenizer = AutoTokenizer.from_pretrained(model_id_to_use)
            
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        logger.info(f"Loading 4-bit quantized causal LM: {model_id_to_use}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id_to_use,
            quantization_config=bnb_config,
            device_map="auto",
            token=HF_TOKEN or None if model_id_to_use == DEFAULT_MODEL_ID else None
        )
        model = prepare_model_for_kbit_training(model)
        
        # LoRA Configuration (Targeting all linear attention projections)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            optim="paged_adamw_8bit",
            push_to_hub=push_to_hub and bool(HF_TOKEN),
            hub_model_id=self.hub_model_id if push_to_hub and bool(HF_TOKEN) else None,
            hub_token=HF_TOKEN or None,
            report_to="none"
        )
        
        # Format dataset
        def format_batch(batch):
            return [self.format_prompt({"question": q, "answer": a}) for q, a in zip(batch["question"], batch["answer"])]
            
        logger.info("Starting SFTTrainer QLoRA fine-tuning loop...")
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            formatting_func=format_batch,
            max_seq_length=512
        )
        
        trainer.train()
        
        # Save local adapter
        final_adapter_dir = self.output_dir / "final_adapter"
        trainer.model.save_pretrained(final_adapter_dir)
        tokenizer.save_pretrained(final_adapter_dir)
        logger.info(f"Saved local LoRA adapter to {final_adapter_dir}")
        
        if push_to_hub and HF_TOKEN:
            logger.info(f"Pushing model adapter to HuggingFace Hub ({self.hub_model_id})...")
            trainer.push_to_hub()
            logger.info("SUCCESS: Model pushed to HuggingFace Hub!")
            
        return str(final_adapter_dir)


if __name__ == "__main__":
    trainer = TradeLLMTrainer()
    if HAS_LLM_LIBS and torch.cuda.is_available():
        out = trainer.train_and_push(num_train_epochs=1, push_to_hub=False)
        print(f"\n[+] QLoRA Fine-tuning Complete: {out}")
    else:
        print("\n[!] PyTorch/GPU not available locally. This module is ready to be executed on Camber/Lightning GPU studio via job_llm.sh!")
