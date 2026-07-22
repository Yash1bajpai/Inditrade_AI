import json
import os

def read_jsonl(filepath):
    """Read a JSONL file and return a list of dictionaries."""
    if not os.path.exists(filepath):
        print("Dataset not found!")
        return None

    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def write_jsonl(filepath, rows):
    """Write a list of dictionaries to a JSONL file."""
    with open(filepath, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
