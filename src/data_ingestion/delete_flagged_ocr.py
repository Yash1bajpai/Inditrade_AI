import json
import os
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.file_io import read_jsonl, write_jsonl

def run_deletion():
    dataset_path = "data/processed/policy_qa_dataset.jsonl"
    all_rows = read_jsonl(dataset_path)
    if all_rows is None:
        return

    c1_deleted_refs = []
    c2_deleted_refs = []
    retained_rows = []

    for row in all_rows:
        if row.get("doc_type") == "DGFT_OCR":
            ans = row.get("answer", "")
            ans_str = ans.strip()

            cond1 = bool(re.search(r'(?:[\[(]?M\-?\d+[\/0-9a-zA-Z\.\-_\(\)]+\]?|\b(?:File|F)\.?\s*No[\.\s]*[\/0-9a-zA-Z\.\-_\(\)]+\]?|\[?\b[0-9]{2,4}\/[0-9]{2,4}\/[0-9a-zA-Z\.\-_\(\)\/]+(?:PC|E\-)[0-9a-zA-Z\.\-_\(\)\/]+\]?)\.?$', ans_str))

            cond2 = bool(re.search(r'\.\.\s*Exporters,\s*importers,\s*and\s*relevant\s*trade\s*authorities\s*are\s*required\s*to\s*comply\s*with\s*these\s*provisions', ans_str))

            if cond1:
                c1_deleted_refs.append(row.get("source_ref_id", "Unknown"))
                continue
            elif cond2:
                c2_deleted_refs.append(row.get("source_ref_id", "Unknown"))
                continue
            else:
                retained_rows.append(row)
        else:
            retained_rows.append(row)

    write_jsonl(dataset_path, retained_rows)

    print("=== DELETION SUMMARY ===")
    print(f"Condition 1 (raw filing code ending) deleted count: {len(c1_deleted_refs)}")
    if c1_deleted_refs:
        print(f"Condition 1 Ref IDs deleted: {', '.join(sorted(set(c1_deleted_refs)))}")

    print(f"\nCondition 2 ('..' followed by boilerplate compliance ending) deleted count: {len(c2_deleted_refs)}")
    if c2_deleted_refs:
        print(f"Condition 2 Unique Ref IDs deleted ({len(set(c2_deleted_refs))} unique docs): {', '.join(sorted(set(c2_deleted_refs))[:40])} ... (see full list on disk)")

    print(f"\nTotal DGFT_OCR entries deleted across both conditions: {len(c1_deleted_refs) + len(c2_deleted_refs)}")
    print(f"New total line count of policy_qa_dataset.jsonl: {len(retained_rows)}")

if __name__ == "__main__":
    run_deletion()

