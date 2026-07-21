import json
import os
import re

def strip_letterhead(text: str) -> str:
    if not text:
        return ""

    cleaned = re.sub(
        r'(?i)(\(?[\[\s]*to\s+be\s+published\s+in\s+the\s+gazette\s+of\s+india.*?vanijya\s+bhawan|to\s+be\s+published\s+in\s+the\s+gazette\s+of\s+india.*?sub\-?section\s*\([i1]+\)|government\s+of\s+india\s+ministry\s+of\s+commerce.*?vanijya\s+bhawan|government\s+of\s+india\s+ministry\s+of\s+commerce.*?directorate\s+general\s+of\s+foreign\s+trade|directorate\s+general\s+of\s+foreign\s+trade.*?vanijya\s+bhawan|directorate\s+general\s+of\s+foreign\s+trade.*?new\s+delhi,\s*dated[:\s0-9a-z{}+_.,*\-]+|\[?\s*notification\s+no[:.\s0-9a-z\/-]+|\[?\s*trade\s+notice\s+no[:.\s0-9a-z\/-]+|\[?\s*public\s+notice\s+no[:.\s0-9a-z\/-]+|vanijya\s+bhawan|new\s+delhi,\s*dated[:\s0-9a-z{}+_.,*\-]+)',
        ' ',
        text
    )
    cleaned = re.sub(r'\[Page\s+\d+\]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def extract_clean_subject_and_body(context_snippet: str) -> tuple:
    clean_text = strip_letterhead(context_snippet)

    subject = ""
    sub_match = re.search(r'(?i)(?:Subject|Sub)\s*:\s*([^\n\r\[]{12,280})', clean_text)
    if sub_match:
        subject = sub_match.group(1).strip()
        subject = re.sub(r'[\s\-:_.,]+$', '', subject)

    if subject:
        sub_split = re.split(r'(?i)(?:Subject|Sub)\s*:', clean_text)
        body = sub_split[-1] if len(sub_split) > 1 else clean_text
        if body.strip().startswith(subject):
            body = body.strip()[len(subject):].strip()
    else:
        body = clean_text

    if not subject and len(body) >= 20:
        words = body.split()
        subject = " ".join(words[:12]) if len(words) >= 12 else body

    subject = strip_letterhead(subject)
    body = strip_letterhead(body)

    return subject.strip(), body.strip()

def regenerate_answer(row: dict, subject: str, body: str) -> str:
    doc_label = "DGFT Notification/Notice"
    ref_id = str(row.get("source_ref_id", "N/A"))
    date_str = str(row.get("source_date", "N/A"))
    question = row.get("question", "")

    operative_clauses = body
    if len(operative_clauses) < 20 and subject:
        operative_clauses = subject

    operative_clauses = re.sub(r'^[\])|:;,\-.\s]+', '', operative_clauses).strip()
    subject = re.sub(r'^[\])|:;,\-.\s]+', '', subject).strip()

    if "primary subject and policy mandate" in question:
        if subject and len(subject) > 12:
            return f"According to {doc_label} {ref_id} (dated {date_str}), the primary policy subject is: {subject}. Specifically, the notification establishes regulations and mandates regarding: {operative_clauses[:260]}."
        else:
            return f"According to {doc_label} {ref_id} (dated {date_str}), the document issues trade policy instructions and mandates regarding: {operative_clauses[:280]}."
    elif "specific procedural guidelines, schedule conditions, or item details" in question:
        return f"Under {doc_label} {ref_id}, the specific conditions, item classifications, and procedural rules specified are: {operative_clauses[:280]}. Exporters, importers, and trade authorities must strictly comply with these revised schedule conditions."
    elif "effective timeline and legal authority" in question:
        return f"{doc_label} {ref_id} is officially issued under the authority of the Directorate General of Foreign Trade, dated {date_str}. The provisions (`{subject[:150]}`) take effect in accordance with the formal trade notification timelines."
    else:
        return f"As notified under {doc_label} {ref_id} ({date_str}), the policy requirements are: {operative_clauses[:280]}."

def run_cleanup():

    dataset_path = "data/processed/policy_qa_dataset.jsonl"
    if not os.path.exists(dataset_path):
        print("Dataset not found!")
        return

    all_rows = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                all_rows.append(json.loads(l))

    flagged_count = 0
    regenerated_count = 0
    deleted_count = 0

    keywords = ["Gazette of India", "Vanijya Bhawan", "Government of India Ministry of Commerce"]

    updated_rows = []
    for row in all_rows:
        if row.get("doc_type") == "DGFT_OCR" and any(kw in row.get("answer", "") for kw in keywords):
            flagged_count += 1

            cs = row.get("context_snippet", "")
            subject, body = extract_clean_subject_and_body(cs)

            if any(kw.lower() in subject.lower() for kw in keywords) or any(kw.lower() in body.lower() for kw in keywords):
                for kw in keywords:
                    subject = re.sub(re.escape(kw), ' ', subject, flags=re.IGNORECASE)
                    body = re.sub(re.escape(kw), ' ', body, flags=re.IGNORECASE)
                subject = re.sub(r'\s+', ' ', subject).strip()
                body = re.sub(r'\s+', ' ', body).strip()

            total_words = len(re.findall(r'\b[a-zA-Z]{3,}\b', f"{subject} {body}"))
            if total_words < 16 or not subject:
                deleted_count += 1
                continue

            new_ans = regenerate_answer(row, subject, body)

            if any(kw.lower() in new_ans.lower() for kw in keywords):
                deleted_count += 1
                continue

            row["answer"] = new_ans
            regenerated_count += 1
            updated_rows.append(row)
        else:
            updated_rows.append(row)

    with open(dataset_path, "w", encoding="utf-8") as f:
        for r in updated_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Flagged entries count: {flagged_count}")
    print(f"Successfully regenerated with real answer: {regenerated_count}")
    print(f"Deleted as unfixable/fragmentary: {deleted_count}")
    print(f"New total line count of policy_qa_dataset.jsonl: {len(updated_rows)}")

if __name__ == "__main__":
    run_cleanup()

