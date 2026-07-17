# IndiTrade AI - Phase 5: Method 1 QLoRA Fine-Tuning & Hugging Face Free Serverless Inference Runbook

This guide walks you through fine-tuning **Llama-3-8B-Instruct** on our verified **1,360-pair Indian Foreign Trade Policy Q&A dataset (`data/processed/policy_qa_dataset.jsonl`)** on **Lightning AI Studio** (or Google Colab / Kaggle) and deploying it **100% for FREE** via **Hugging Face Free Serverless Inference API (Method 1)**.

---

## 🏗️ Architecture & Flow (Method 1)

```mermaid
graph LR
    Lightning["⚡ Lightning AI Studio<br/>(Train Llama-3-8B with QLoRA & Unsloth)"] -->|Push Adapter & Merged Model| HF["🤗 Hugging Face Hub<br/>(Yash1bajpai/Inditrade-Llama3-8B-Policy)"]
    HF -->|Auto-hosts 100% FREE API| TGI["🌐 HF Free Serverless Inference API<br/>(api-inference.huggingface.co)"]
    TGI <-->|Standard HTTP POST / fetch()| Vercel["⚡ Vercel Web App<br/>(Next.js / Vanilla JS Frontend)"]
```

---

## ⚡ Step-by-Step Training Guide (Lightning AI Studio or Google Colab)

### 1. Setup & Dependencies
Open a terminal inside your GPU workspace (`Lightning AI Studio with free T4/L4 GPU` or `Google Colab free T4`) and run:
```bash
# Clone or ensure you are in the project root
cd Inditrade_AI

# Install specialized Unsloth + QLoRA requirements
pip install -r src/models/requirements_finetune.txt
```

### 2. Login to Hugging Face Hub
To download the Llama-3 base model and push your fine-tuned model for Free Inference, log in:
```bash
huggingface-cli login
# Paste your Hugging Face Access Token with 'Write' permissions when prompted
```

### 3. Dry-Run Verification (Test Pipeline in 2 Minutes)
Before running a full 3-epoch training loop, run a 50-step dry run to verify dataset formatting, memory usage, and checkpoint creation:
```bash
python src/models/fine_tune_unsloth.py --max-steps 50 --batch-size 2
```
*Expected Output*: Saves test LoRA adapter weights to `models/checkpoints/inditrade-llama3-8b-policy-lora`.

### 4. Full Production Training & Push to Hugging Face Hub (`Method 1`)
Run the full 3-epoch fine-tuning pass across all 1,360 Q&A pairs and automatically push both the LoRA adapter and the 16-bit standalone merged model to Hugging Face Hub so your Free Serverless Inference API endpoint goes live:
```bash
python src/models/fine_tune_unsloth.py \
    --epochs 3 \
    --batch-size 2 \
    --grad-accum 4 \
    --lr 2e-4 \
    --push-to-hub \
    --push-merged-16bit \
    --hub-model-id Yash1bajpai/Inditrade-Llama3-8B-Policy \
    --hf-token YOUR_HF_TOKEN
```
*Training Time*: Approximately **18 to 22 minutes** on a T4/L4 GPU using Unsloth.

---

## 🧪 Verifying Your Free Serverless Inference API

Once the model is pushed to `https://huggingface.co/Yash1bajpai/Inditrade-Llama3-8B-Policy-Merged-16Bit`, Hugging Face automatically spins up the Free Serverless Inference endpoint.

Test your live cloud endpoint immediately from any terminal using `test_hf_inference.py`:
```bash
python src/models/test_hf_inference.py \
    --mode cloud \
    --model-id Yash1bajpai/Inditrade-Llama3-8B-Policy-Merged-16Bit \
    --hf-token YOUR_HF_TOKEN
```
*(Note: If you receive an HTTP 503 error on the first query, it simply means Hugging Face is loading your model into memory from cold storage. Wait 30–45 seconds and re-run.)*

---

## 🌐 Connecting to Your Vercel Frontend (`Phase 6 Preview`)

Inside your Vercel Next.js application (`or serverless function / api route`), query your model with zero external costs:

```javascript
// pages/api/chat.js or app/api/chat/route.js
export async function POST(req) {
  const { question, context } = await req.json();

  const prompt = `### Instruction:\nYou are an expert Indian Foreign Trade Policy assistant for IndiTrade AI. Answer accurately.\n\n### Context:\n${context || "DGFT Regulatory Framework"}\n\n### Question:\n${question}\n\n### Answer:\n`;

  const response = await fetch(
    "https://api-inference.huggingface.co/models/Yash1bajpai/Inditrade-Llama3-8B-Policy-Merged-16Bit",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${process.env.HF_TOKEN}`
      },
      body: JSON.stringify({
        inputs: prompt,
        parameters: {
          max_new_tokens: 300,
          temperature: 0.2,
          top_p: 0.9,
          return_full_text: false
        }
      })
    }
  );

  const data = await response.json();
  const answer = data[0]?.generated_text || "Unable to generate answer at this time.";
  return new Response(JSON.stringify({ answer }), { status: 200 });
}
```
