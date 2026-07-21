import json
import re
from collections import Counter

def check_coherence(ans: str) -> list:
    reasons = []
    if not ans:
        return ["Empty answer"]

    if re.search(r'(?i)\b(subject\s*[:\-]|s\.o\.\s*\(e\)|new\s+delhi,\s*dated|to\s+be\s+published|notification\s+no[:\.]|trade\s+notice\s+no[:\.]|public\s+notice\s+no[:\.]|\[page\s*\d+\]|to,\s*1\.\s*all|part\-?ii,\s*section|extraordinary\s+part)', ans):
        reasons.append("Structural fragment (Subject/Dated/S.O./Header)")

    if re.search(r'\b0\d{2,}\s+[A-Za-z]|\d+[\"\'\`]\s*[A-Za-z]|[{[|]\d+\s+[A-Za-z]', ans) or re.search(r'([a-zA-Z]\d[a-zA-Z]|\b[a-z]{1,2}[?|_~][a-z]{1,2}\b)', ans):
        reasons.append("OCR garble/digit-garble")

    if ans.strip().endswith(('...', '(', '[', '-', ':', ',', ' of', ' for', ' under', ' in', ' to', ' and', ' the', ' a', ' an', ' or', ' with', ' by')):
        reasons.append("Mid-sentence cutoff / trailing fragment")

    words = re.findall(r'\b[a-zA-Z]{2,}\b', ans)
    if len(words) < 12 or not re.search(r'[a-zA-Z0-9][.!?](?:\s|$)', ans):
        reasons.append("No coherent sentence / too short")

    if re.search(r'\b(?:Sl\s*No|HS\s*Code|Item\s*Description|Policy\s*Condition|Existing\s*Policy|Revised\s*Policy)\b.*\b(?:Sl\s*No|HS\s*Code|Item\s*Description|Policy\s*Condition|Existing\s*Policy|Revised\s*Policy)\b', ans, re.IGNORECASE):
        reasons.append("Table header fragment dump")

    return reasons

def run_inspection():
    qa = [json.loads(l) for l in open("data/processed/policy_qa_dataset.jsonl", encoding="utf-8") if l.strip()]
    ocr = [r for r in qa if r.get("doc_type") == "DGFT_OCR"]
    print(f"Total DGFT_OCR entries: {len(ocr)}")

    flagged = []
    for r in ocr:
        reasons = check_coherence(r.get("answer", ""))
        if reasons:
            flagged.append((r, reasons))

    print(f"Total flagged by general coherence check: {len(flagged)}")
    all_reasons = [item for _, rlist in flagged for item in rlist]
    print("Reasons breakdown:", Counter(all_reasons))
    print("\n--- Sample Flagged Entries ---")
    for i in range(min(8, len(flagged))):
        print(f"[{flagged[i][0]['source_ref_id']}] Reasons: {flagged[i][1]}")
        print("   Ans:", flagged[i][0]['answer'][:200])
        print()

if __name__ == "__main__":
    run_inspection()

