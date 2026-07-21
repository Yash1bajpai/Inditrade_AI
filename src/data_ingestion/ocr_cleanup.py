"""
OCR Cleanup Utility (`ocr_cleanup.py`).

Performs regex and heuristic cleanup on `data/processed/dgft_ocr_chunks.jsonl`
to normalize typographic quotes/dashes and strip garbled non-ASCII-adjacent runs
or Hindi text misread by English-only Tesseract OCR.
"""

import json
import re
import os
import sys

sys_reconfig = getattr(sys.stdout, 'reconfigure', None)
if sys_reconfig:
    sys_reconfig(encoding='utf-8', errors='replace')

def clean_ocr_text(text: str) -> str:

    replacements = {
        '‘': "'", '’': "'", '“': '"', '”': '"',
        '—': '-', '–': '-', '…': '...', '°': ' deg ',
        '\ufffd': ' '
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    text = re.sub(r'([!@#$%^&*~|_+=<>/]){3,}', ' ', text)

    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_ocr_cleanup(jsonl_path: str = "data/processed/dgft_ocr_chunks.jsonl"):
    print(f"=== IndiTrade AI: OCR Heuristic & Regex Cleanup ===")
    if not os.path.exists(jsonl_path):
        print(f"[ERROR] File not found: {jsonl_path}")
        return

    with open(jsonl_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    print(f"Loaded {len(lines)} chunks from `{jsonl_path}`. Cleaning non-ASCII runs...")

    cleaned_chunks = []
    chars_removed_total = 0

    for line in lines:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            old_text = data["clean_text"]
            new_text = clean_ocr_text(old_text)
            chars_removed_total += (len(old_text) - len(new_text))
            data["clean_text"] = new_text
            cleaned_chunks.append(data)
        except Exception as e:
            print(f"  [WARNING] Could not parse chunk line: {e}")

    with open(jsonl_path, "w", encoding="utf-8") as out_f:
        for c in cleaned_chunks:
            out_f.write(json.dumps(c, ensure_ascii=True) + "\n")

    print(f"  ✅ Successfully cleaned and updated {len(cleaned_chunks)} OCR chunks!")
    print(f"  ✅ Total garbled/non-ASCII characters stripped across corpus: {chars_removed_total:,}")
    print("===================================================\n")

if __name__ == "__main__":
    run_ocr_cleanup()

