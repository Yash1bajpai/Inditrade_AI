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
        raw_dataset = load_dataset("json", data_files=str(self.dataset_path), split="train")
        
        def add_text_col(ex):
            if isinstance(ex.get("question"), list):
                return {"text": [self.format_prompt({"question": q, "answer": a}) for q, a in zip(ex["question"], ex["answer"])]}
            else:
                return {"text": self.format_prompt(ex)}
            
        dataset = raw_dataset.map(add_text_col)
        
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
            if "tinyllama" in model_id_to_use.lower():
                self.hub_model_id = "yashbajpai/inditrade-tinyllama-1.1b"
            
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        logger.info(f"Loading 4-bit quantized causal LM: {model_id_to_use} (enforcing float16 for T4 GPU compatibility)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id_to_use,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,  # Required for Tesla T4 (Turing arch does not support bfloat16 grad scaling)
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
        
        try:
            from trl import SFTConfig
            config_class = SFTConfig
        except ImportError:
            config_class = TrainingArguments
            
        import inspect
        config_params = inspect.signature(config_class.__init__).parameters
        config_kwargs = {
            "output_dir": str(self.output_dir),
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "fp16": True,
            "bf16": False,  # Explicitly disable bf16 for Tesla T4 GPU compatibility
            "logging_steps": 10,
            "save_strategy": "epoch",
            "optim": "paged_adamw_8bit",
            "push_to_hub": False,  # Disabled during training to prevent 403 Forbidden crash on read-only tokens
            "report_to": "none"
        }
        if "max_length" in config_params:
            config_kwargs["max_length"] = 512
        elif "max_seq_length" in config_params:
            config_kwargs["max_seq_length"] = 512
            
        if "dataset_text_field" in config_params:
            config_kwargs["dataset_text_field"] = "text"
            
        training_args = config_class(**config_kwargs)
            
        logger.info("Starting SFTTrainer QLoRA fine-tuning loop...")
        sft_params = inspect.signature(SFTTrainer.__init__).parameters
        sft_kwargs = {
            "model": model,
            "args": training_args,
            "train_dataset": dataset,
        }
        if "dataset_text_field" in sft_params and "dataset_text_field" not in config_kwargs:
            sft_kwargs["dataset_text_field"] = "text"
            
        if "max_length" in sft_params and "max_length" not in config_kwargs and "max_seq_length" not in config_kwargs:
            sft_kwargs["max_length"] = 512
        elif "max_seq_length" in sft_params and "max_length" not in config_kwargs and "max_seq_length" not in config_kwargs:
            sft_kwargs["max_seq_length"] = 512
            
        if "processing_class" in sft_params:
            sft_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in sft_params:
            sft_kwargs["tokenizer"] = tokenizer
        else:
            sft_kwargs["processing_class"] = tokenizer
            
        trainer = SFTTrainer(**sft_kwargs)
        
        trainer.train()
        
        # Save local adapter
        final_adapter_dir = self.output_dir / "final_adapter"
        trainer.model.save_pretrained(final_adapter_dir)
        tokenizer.save_pretrained(final_adapter_dir)
        logger.info(f"Saved local LoRA adapter to {final_adapter_dir}")
        
        # Safely attempt push to HuggingFace Hub without crashing if token is read-only
        if push_to_hub and HF_TOKEN and self.hub_model_id:
            try:
                logger.info(f"Attempting to push trained LoRA adapter to HuggingFace Hub ({self.hub_model_id})...")
                trainer.model.push_to_hub(self.hub_model_id, token=HF_TOKEN)
                tokenizer.push_to_hub(self.hub_model_id, token=HF_TOKEN)
                logger.info("SUCCESS: Pushed LoRA adapter to HuggingFace Hub!")
            except Exception as e:
                logger.warning(f"Could not push to HuggingFace Hub (token lacks write permissions / 403 Forbidden): {e}. Model is safely saved locally at {final_adapter_dir}!")
            
        return str(final_adapter_dir)


if __name__ == "__main__":
    trainer = TradeLLMTrainer()
    if HAS_LLM_LIBS and torch.cuda.is_available():
        out = trainer.train_and_push(num_train_epochs=1, push_to_hub=False)
        print(f"\n[+] QLoRA Fine-tuning Complete: {out}")
    else:
        print("\n[!] PyTorch/GPU not available locally. This module is ready to be executed on Camber/Lightning GPU studio via job_llm.sh!")
