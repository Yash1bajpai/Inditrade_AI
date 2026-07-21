import json
import os
import re
import random
from collections import Counter

def check_coherence(ans: str) -> list:
    reasons = []
    if not ans or len(ans.strip()) < 40:
        return ["Empty or too short answer"]

    if re.search(r'(?i)\b(s\.o\.\s*\(e\)|new\s+delhi,\s*dated|to\s+be\s+published|notification\s+no[:\.]|trade\s+notice\s+no[:\.]|public\s+notice\s+no[:\.]|\[page\s*\d+\]|to,\s*1\.\s*all|part\-?ii,\s*section|extraordinary\s+part|subject\s*:\s*\-|in\s+exercise\s+of\s+powers\s+conferred)', ans):
        reasons.append("Structural fragment (Subject/Dated/S.O./Header/Preamble)")

    if re.search(r'\b018\s+[A-Za-z]|\d+[\"\'\`\*]+\s+[A-Za-z]|[{[|]\d+\s+[A-Za-z]', ans) or re.search(r'([a-zA-Z]\d[a-zA-Z]|\b[a-z]{1,2}[?|_~][a-z]{1,2}\b)', ans):
        reasons.append("OCR garble/digit-garble")

    if ans.strip().endswith(('...', '(', '[', '-', ':', ',', ' of', ' for', ' under', ' in', ' to', ' and', ' the', ' a', ' an', ' or', ' with', ' by', ' regarding', ' th', ' th.', ' of.', ' the.')):
        reasons.append("Mid-sentence cutoff / trailing fragment")

    if re.search(r'\b(?:Sl\s*No|HS\s*Code|Item\s*Description|Policy\s*Condition|Existing\s*Policy|Revised\s*Policy)\b.*\b(?:Sl\s*No|HS\s*Code|Item\s*Description|Policy\s*Condition|Existing\s*Policy|Revised\s*Policy)\b', ans, re.IGNORECASE):
        reasons.append("Table header fragment dump")

    return reasons

def fix_ocr_garble(text: str) -> str:
    if not text:
        return ""
    t = text

    t = re.sub(r'\b018\s+May\b', '01 May', t, flags=re.IGNORECASE)
    t = re.sub(r'\b(\d{1,2})[\"\'\`\*]+\s+([A-Z][a-z]+)', r'\1 \2', t)
    t = re.sub(r'\b(\d{1,2})\s*\+\s*[a-z*]+\s+([A-Z][a-z]+)', r'\1 \2', t)
    t = re.sub(r'[{[|]\s*(\d{1,2})\s+([A-Z][a-z]+)', r'\1 \2', t)

    t = re.sub(r'(?i)\b(?:Sl\.?\s*No\.?|HS\s*Code|Item\s*Description|Import\s*Policy|Policy\s*Condition|Existing\s*Policy|Revised\s*Policy)\b[\s|:\-]*', ' ', t)

    t = re.sub(r'(?i)\bS\.?O\.?\s*\(E\)\.?[\s:\-]*', ' ', t)
    t = re.sub(r'(?i)\bIn\s+exercise\s+of\s+powers\s+conferred\b.*$', ' ', t)

    t = re.sub(r'\[Page\s+\d+\]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def extract_clean_subject(context_snippet: str) -> str:
    cs = fix_ocr_garble(context_snippet)
    sub_match = re.search(r'(?i)(?:Subject|Sub)\s*:\s*([^\n\r\[]{12,280})', cs)
    if sub_match:
        sub = sub_match.group(1).strip()
        sub = re.sub(r'(?i)\b(?:S\.O\.\s*\(E\)|In\s+exercise\s+of\s+powers|New\s+Delhi).*$', '', sub).strip()
        sub = re.sub(r'[\s\-:_.,()\[\]]+$', '', sub).strip()
        sub = re.sub(r'^[()\[\]\-:_.,\s]+', '', sub).strip()
        if len(sub) >= 12:
            return sub

    cs_clean = re.sub(r'(?i)(to\s+be\s+published.*?vanijya\s+bhawan|government\s+of\s+india.*?vanijya\s+bhawan|directorate\s+general.*?new\s+delhi,\s*dated[:\s0-9a-z\-]+|notification\s+no[:\.\s0-9a-z\/-]+)', ' ', cs)
    cs_clean = fix_ocr_garble(cs_clean)
    words = cs_clean.split()
    if len(words) >= 10:
        sub = " ".join(words[:16]).strip(' .,:-()[]')
        return sub
    return cs_clean[:120].strip(' .,:-()[]')

def extract_clean_operative_summary(context_snippet: str, subject: str) -> str:
    cs = fix_ocr_garble(context_snippet)
    sub_pos = cs.lower().find("subject:")
    if sub_pos != -1:
        body = cs[sub_pos + len("subject:"):].strip()
        if body.lower().startswith(subject.lower()):
            body = body[len(subject):].strip()
    else:
        body = cs

    body = fix_ocr_garble(body)
    body = re.sub(r'^[\])|:;,\-.\s()]+', '', body).strip()

    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', body)
    valid_sentences = [
        s.strip() for s in sentences
        if len(s.strip()) >= 25
        and not s.strip().lower().startswith(('to,', 'all exporters', 'copy to', 'in exercise of powers'))
        and not s.strip().endswith((' of', ' for', ' under', ' in', ' to', ' and', ' the', ' th', ' th.', ' of.'))
    ]

    if valid_sentences:
        summary = " ".join(valid_sentences[:2]).strip()
        if not summary.endswith('.'):
            summary += "."
        return summary
    return ""

def synthesize_clean_answer(row: dict) -> str:
    ref_id = str(row.get("source_ref_id", "N/A"))
    date_str = str(row.get("source_date", "N/A"))
    question = row.get("question", "")
    cs = row.get("context_snippet", "")

    subject = extract_clean_subject(cs)
    operative = extract_clean_operative_summary(cs, subject)

    if "primary subject and policy mandate" in question:
        if operative and len(operative.split()) >= 10:
            ans = f"According to DGFT Notification {ref_id} (dated {date_str}), the primary policy subject is: {subject}. Specifically, the notification establishes regulatory rules and provisions: {operative}"
        else:
            ans = f"According to DGFT Notification {ref_id} (dated {date_str}), the primary policy subject and mandate is: {subject}. The notification officially establishes regulatory procedures and conditions governing this trade policy revision."
    elif "specific procedural guidelines, schedule conditions, or item details" in question:
        if operative and len(operative.split()) >= 12:
            ans = f"Under DGFT Notification {ref_id}, the specific procedural rules, item classifications, and schedule conditions apply as follows: {operative} Exporters, importers, and customs authorities are required to strictly adhere to these updated regulatory conditions."
        else:

            return ""
    elif "effective timeline and legal authority" in question:
        ans = f"DGFT Notification {ref_id} is officially issued under the legal authority of the Directorate General of Foreign Trade, dated {date_str}. The operative provisions regarding `{subject}` take effect in accordance with the formal regulatory timelines established by the Ministry of Commerce & Industry."
    else:
        if operative and len(operative.split()) >= 10:
            ans = f"As notified under DGFT Notification {ref_id} ({date_str}), the regulatory provisions and policy mandates are: {operative}"
        else:
            ans = f"As notified under DGFT Notification {ref_id} ({date_str}), the official trade policy instruction governs the requirements and procedures for: {subject}."

    ans = re.sub(r'\.\.\.+$', '.', ans)
    ans = re.sub(r'\s+', ' ', ans).strip()
    if ans and not ans.endswith('.'):
        ans += "."
    return ans

def run_coherence_regeneration():

    dataset_path = "data/processed/policy_qa_dataset.jsonl"
    if not os.path.exists(dataset_path):
        print("Dataset not found!")
        return

    all_rows = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                all_rows.append(json.loads(l))

    flagged_entries = []
    updated_rows = []
    deleted_count = 0
    regenerated_count = 0

    for row in all_rows:
        if row.get("doc_type") == "DGFT_OCR":
            ans = row.get("answer", "")
            reasons = check_coherence(ans)
            if reasons:
                cs = row.get("context_snippet", "")
                subject = extract_clean_subject(cs)

                clean_cs_words = re.findall(r'\b[a-zA-Z]{3,}\b', fix_ocr_garble(cs))
                if len(clean_cs_words) < 12 or not subject or len(subject.split()) < 3:
                    deleted_count += 1
                    continue

                new_ans = synthesize_clean_answer(row)
                if not new_ans:
                    deleted_count += 1
                    continue

                new_reasons = check_coherence(new_ans)
                if new_reasons:
                    deleted_count += 1
                    continue

                row["answer"] = new_ans
                regenerated_count += 1
                flagged_entries.append((row, reasons))
                updated_rows.append(row)
            else:
                updated_rows.append(row)
        else:
            updated_rows.append(row)

    with open(dataset_path, "w", encoding="utf-8") as f:
        for r in updated_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Total DGFT_OCR answers flagged by coherence check: {len(flagged_entries) + deleted_count}")
    print(f"Successfully regenerated with clean synthesized answer: {regenerated_count}")
    print(f"Deleted as unfixable/fragmentary context: {deleted_count}")
    print(f"New total line count of policy_qa_dataset.jsonl: {len(updated_rows)}")

    print("\n========================================================")
    print("5 RANDOM FULL REGENERATED ANSWERS FOR SPOT-CHECKING:")
    print("========================================================")

    sample_size = min(5, len(flagged_entries))
    if sample_size > 0:
        samples = random.sample(flagged_entries, sample_size)
        for idx, (s_row, orig_reasons) in enumerate(samples, 1):
            print(f"\n[SPOT-CHECK #{idx}] | Ref: {s_row['source_ref_id']} | Date: {s_row.get('source_date', 'N/A')}")
            print(f"ORIGINAL FLAGGED REASONS: {orig_reasons}")
            print(f"QUESTION:\n{s_row['question']}")
            print(f"FULL REGENERATED ANSWER:\n{s_row['answer']}")
            print("-" * 75)

if __name__ == "__main__":
    run_coherence_regeneration()

