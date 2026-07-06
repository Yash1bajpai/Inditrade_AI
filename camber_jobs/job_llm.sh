#!/bin/bash
# Export Lightning AI / Camber main environment to PATH
export PATH=/opt/jupyter/envs/main/bin:$PATH
# ==========================================
# Camber / Lightning AI Job — Llama-3.2-1B QLoRA
# ==========================================
# Executes 4-bit QLoRA fine-tuning on Indian trade policy Q&A dataset.
# Target: GPU Engine (8 CPU, 32GB RAM, 1x NVIDIA GPU / T4 / A10G / V100) (~15 GPU hours)

echo "[+] Starting Llama-3.2-1B QLoRA Fine-tuning Job on Cloud GPU..."
nvidia-smi

echo "[+] Installing deep learning & PEFT dependencies..."
pip install torch transformers peft accelerate bitsandbytes datasets trl huggingface_hub python-dotenv --quiet

echo "[+] Verifying Q&A dataset..."
python -c "
from pathlib import Path
if not Path('data/processed/policy_qa_dataset.jsonl').exists():
    print('[!] Q&A dataset not found! Running synthetic Q&A generator first...')
    from src.data.generate_qa_dataset import SyntheticQAGenerator
    SyntheticQAGenerator().run_pipeline()
"

echo "[+] Starting SFTTrainer QLoRA training loop..."
python -c "
from src.models.llm_qlora import TradeLLMTrainer
trainer = TradeLLMTrainer()
# Trains for 3 epochs and pushes to HuggingFace Hub (yashbajpai/inditrade-llama-3.2-1b)
out = trainer.train_and_push(num_train_epochs=3, push_to_hub=True)
print(f'\n[✓] QLoRA Job Complete! Adapter saved at: {out}')
"
