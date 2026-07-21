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
    
    mocs = [
        ("MOC_data_ingestion", "MOC_Hub", json.dumps({"description": "Module B: Multimodal Policy Ingestion & Q&A Dataset Generation Hub"})),
        ("MOC_models", "MOC_Hub", json.dumps({"description": "Module C & Module E: AI Forecasting, Anomaly Detection & LLM Fine-Tuning Hub"}))
    ]
    for m_id, label, props in mocs:
        cursor.execute("INSERT OR REPLACE INTO Nodes (id, label, properties, created_at, updated_at, trust_score, is_deleted) VALUES (?, ?, ?, ?, ?, 1.0, 0)", (m_id, label, props, now_iso, now_iso))
        
    nodes_to_add = [
        ("File_direct_local_generator.py", "File", json.dumps({"path": "src/data_ingestion/direct_local_generator.py", "description": "High-throughput local policy Q&A generator covering 100% of digital, OCR, and PIB documents."})),
        ("File_clean_ocr_answers.py", "File", json.dumps({"path": "src/data_ingestion/clean_ocr_answers.py", "description": "Strips raw document letterhead and headers from OCR Q&A pairs."})),
        ("File_regenerate_ocr_coherence.py", "File", json.dumps({"path": "src/data_ingestion/regenerate_ocr_coherence.py", "description": "Enforces strict sentence coherence, removes legal boilerplate preambles, and fixes OCR date garble."})),
        ("File_delete_flagged_ocr.py", "File", json.dumps({"path": "src/data_ingestion/delete_flagged_ocr.py", "description": "Deletes specific filing code endings and double-period boilerplate compliance entries."})),
        ("DataStore_policy_qa_dataset.jsonl", "Data_Store", json.dumps({"path": "data/processed/policy_qa_dataset.jsonl", "description": "Finalized 1360-pair high-precision instruction tuning dataset across DGFT & PIB policy docs."})),
        ("File_fine_tune_llm.py", "File", json.dumps({"path": "src/models/fine_tune_llm.py", "description": "Phase 5 LoRA/QLoRA Instruction Fine-Tuning Script using Unsloth & Llama-3-8B-Instruct."}))
    ]
    
    for n_id, label, props in nodes_to_add:
        cursor.execute("INSERT OR REPLACE INTO Nodes (id, label, properties, created_at, updated_at, trust_score, is_deleted) VALUES (?, ?, ?, ?, ?, 1.0, 0)", (n_id, label, props, now_iso, now_iso))
        
    edges_to_add = [
        ("File_direct_local_generator.py", "MOC_data_ingestion", "belongs_to"),
        ("File_clean_ocr_answers.py", "MOC_data_ingestion", "belongs_to"),
        ("File_regenerate_ocr_coherence.py", "MOC_data_ingestion", "belongs_to"),
        ("File_delete_flagged_ocr.py", "MOC_data_ingestion", "belongs_to"),
        ("File_direct_local_generator.py", "DataStore_policy_qa_dataset.jsonl", "generates"),
        ("File_clean_ocr_answers.py", "DataStore_policy_qa_dataset.jsonl", "cleans"),
        ("File_regenerate_ocr_coherence.py", "DataStore_policy_qa_dataset.jsonl", "refines"),
        ("File_delete_flagged_ocr.py", "DataStore_policy_qa_dataset.jsonl", "filters"),
        ("File_fine_tune_llm.py", "MOC_models", "belongs_to"),
        ("File_fine_tune_llm.py", "DataStore_policy_qa_dataset.jsonl", "consumes")
    ]
    
    for src_id, tgt_id, rel in edges_to_add:
        cursor.execute("SELECT 1 FROM Edges WHERE source_id=? AND target_id=? AND relation_type=?", (src_id, tgt_id, rel))
        if not cursor.fetchone():
            cursor.execute("INSERT INTO Edges (source_id, target_id, relation_type, created_at) VALUES (?, ?, ?, ?)", (src_id, tgt_id, rel, now_iso))
            
    db.commit()
    db.close()
    print("Successfully updated Epistemic Graph Memory with Phase 4 & Phase 5 nodes and edges!")

if __name__ == "__main__":
    update_graph()
