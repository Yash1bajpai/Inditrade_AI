import sqlite3
import json
import os
from datetime import datetime

DB_PATH = ".agents/graph_memory.sqlite"

def update_graph():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return
        
    db = sqlite3.connect(DB_PATH)
    cursor = db.cursor()
    now_iso = datetime.now().isoformat() + "Z"
    
    # Ensure MOC_models exists
    cursor.execute("INSERT OR IGNORE INTO Nodes (id, label, properties, trust_score, created_at, updated_at, is_deleted) VALUES (?, ?, ?, 1.0, ?, ?, 0)", 
                   ("MOC_models", "MOC_Hub", json.dumps({"description": "Module C & Module E: AI Forecasting, Anomaly Detection & LLM Fine-Tuning Hub"}), now_iso, now_iso))
        
    nodes_to_add = [
        ("File_fine_tune_unsloth.py", "File", json.dumps({"path": "src/models/fine_tune_unsloth.py", "description": "Phase 5 Masterclass Unsloth QLoRA script for 8B model with HF Hub push support."})),
        ("File_llm_qlora.py", "File", json.dumps({"path": "src/models/llm_qlora.py", "description": "Alternative standard QLoRA fine-tuning script targeting Llama-3.2-1B."})),
        ("File_test_hf_inference.py", "File", json.dumps({"path": "src/models/test_hf_inference.py", "description": "Verification script for testing local LoRA checkpoints and Hugging Face Serverless Inference API."})),
        ("File_requirements_finetune.txt", "File", json.dumps({"path": "src/models/requirements_finetune.txt", "description": "Fine-tuning dependencies (Unsloth, bitsandbytes, PEFT, TRL)."})),
        ("File_README_FINETUNE.md", "File", json.dumps({"path": "src/models/README_FINETUNE.md", "description": "Masterclass runbook documenting the Lightning AI fine-tuning process."})),
        ("File_inspect_ocr_coherence.py", "File", json.dumps({"path": "src/data_ingestion/inspect_ocr_coherence.py", "description": "Utility script for debugging DGFT OCR coherence checks."}))
    ]
    
    for n_id, label, props in nodes_to_add:
        cursor.execute("INSERT OR REPLACE INTO Nodes (id, label, properties, trust_score, created_at, updated_at, is_deleted) VALUES (?, ?, ?, 1.0, ?, ?, 0)", (n_id, label, props, now_iso, now_iso))
        
    edges_to_add = [
        ("File_fine_tune_unsloth.py", "MOC_models", "belongs_to"),
        ("File_llm_qlora.py", "MOC_models", "belongs_to"),
        ("File_test_hf_inference.py", "MOC_models", "belongs_to"),
        ("File_requirements_finetune.txt", "MOC_models", "belongs_to"),
        ("File_README_FINETUNE.md", "MOC_models", "belongs_to"),
        ("File_inspect_ocr_coherence.py", "MOC_data_ingestion", "belongs_to"),
        ("File_fine_tune_unsloth.py", "DataStore_policy_qa_dataset.jsonl", "consumes"),
        ("File_llm_qlora.py", "DataStore_policy_qa_dataset.jsonl", "consumes")
    ]
    
    for src_id, tgt_id, rel in edges_to_add:
        cursor.execute("SELECT 1 FROM Edges WHERE source_id=? AND target_id=? AND relation_type=?", (src_id, tgt_id, rel))
        if not cursor.fetchone():
            cursor.execute("INSERT INTO Edges (source_id, target_id, relation_type, created_at) VALUES (?, ?, ?, ?)", (src_id, tgt_id, rel, now_iso))
            
    # Also delete the old mismatched node if it exists
    cursor.execute("DELETE FROM Nodes WHERE id = 'File_fine_tune_llm.py'")
    cursor.execute("DELETE FROM Edges WHERE source_id = 'File_fine_tune_llm.py' OR target_id = 'File_fine_tune_llm.py'")

    db.commit()
    db.close()
    print("Successfully updated Epistemic Graph Memory with Phase 5 fine-tuning suite nodes!")

if __name__ == "__main__":
    update_graph()
