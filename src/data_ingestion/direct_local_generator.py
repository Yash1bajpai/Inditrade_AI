import json
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple

def clean_text_snippet(text: str, max_len: int = 500) -> str:
    if not text:
        return ""
    cleaned = re.sub(r'\[Page\s+\d+\]', ' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned[:max_len]

def extract_subject_and_topic(clean_text: str, doc_type: str, fallback_title: str) -> Tuple[str, str]:
    subject = fallback_title
    sub_match = re.search(r'(?:Subject|Sub)\s*:\s*([^\n\r\[]{15,250})', clean_text, re.IGNORECASE)
    if sub_match:
        subject = sub_match.group(1).strip()
        subject = re.sub(r'[\s\-:_.,]+$', '', subject)
    
    topic = "trade policy conditions and procedures"
    hs_match = re.search(r'\b(Chapter\s*-\s*\d+|Chapter\s*\d+|Appendix\s*-\s*[0-9A-Z]+|Appendix\s*[0-9A-Z]+|SION\s*[0-9A-Z\-]+|HS\s*Code\s*\d+|SIMS|RoDTEP|Advance\s*Authorisation|EPCG|Interest\s*Subvention|Import\s*Policy|Export\s*Policy)\b', clean_text, re.IGNORECASE)
    if hs_match:
        topic = hs_match.group(1).strip()
    elif len(subject) > 10:
        words = subject.split()
        if len(words) > 4:
            topic = " ".join(words[:6])
        else:
            topic = subject
            
    return subject, topic

def extract_operative_sentences(clean_text: str) -> List[str]:
    text = re.sub(r'\[Page\s+\d+\]', ' ', clean_text)
    text = re.sub(r'\s+', ' ', text)
    
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9\[(])', text)
    operative = []
    
    keywords = [
        'hereby', 'notified', 'amend', 'substituted', 'inserted', 'policy condition',
        'schedule', 'chapter', 'appendix', 'sion', 'effective', 'authorisation',
        'export of', 'import of', 'prohibited', 'restricted', 'free', 'interest subvention',
        'application', 'extension', 'last date', 'directed', 'decided', 'clarified',
        'mandatory', 'operational', 'customs', 'tariff', 'portal', 'online'
    ]
    
    for s in raw_sentences:
        s_clean = s.strip()
        if len(s_clean) < 30 or len(s_clean) > 500:
            continue
        if any(kw in s_clean.lower() for kw in keywords) or re.search(r'\b\d+\/\d+\b|\b202[0-9]\b', s_clean):
            operative.append(s_clean)
            
    if not operative and raw_sentences:
        for s in raw_sentences[1:6]:
            if len(s.strip()) >= 30:
                operative.append(s.strip())
                
    return operative[:6]

def is_garbled_ocr(clean_text: str) -> bool:
    if not clean_text or len(clean_text.strip()) < 100:
        return True
        
    text = clean_text.strip()
    alnum_chars = sum(1 for c in text if c.isalnum() or c.isspace() or c in '.,:-()/')
    if (alnum_chars / max(len(text), 1)) < 0.65:
        return True
        
    if re.search(r'([|?#:*_~]{4,})', text):
        return True
        
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    if len(words) < 12:
        return True
        
    return False

def generate_qa_for_chunk(chunk: Dict[str, Any], doc_type: str, idx_start: int = 1) -> List[Dict[str, Any]]:
    source_ref_id = str(chunk.get('notification_no', chunk.get('prid', chunk.get('chunk_id', 'REF'))))
    clean_text = chunk.get('clean_text', '')
    source_date = chunk.get('date', 'N/A')
    
    if doc_type == "DGFT_OCR":
        doc_label = "DGFT Notification/Notice"
        fallback_title = f"{chunk.get('type', 'DGFT Notification')} No. {source_ref_id}"
    elif doc_type == "DGFT_POLICY":
        doc_label = "DGFT Trade Policy Document"
        fallback_title = chunk.get('subject', f"Notification No. {source_ref_id}")
    else:
        doc_label = "PIB Press Release"
        fallback_title = chunk.get('title', f"Press Release ID {source_ref_id}")
        
    subject, topic = extract_subject_and_topic(clean_text, doc_type, fallback_title)
    operative_sentences = extract_operative_sentences(clean_text)
    
    if operative_sentences:
        summary_p1 = " ".join(operative_sentences[:2])
        summary_p2 = " ".join(operative_sentences[2:4]) if len(operative_sentences) > 2 else operative_sentences[0]
    else:
        clean_snip = re.sub(r'\s+', ' ', clean_text[:800]).strip()
        summary_p1 = clean_snip
        summary_p2 = clean_snip
        
    snippet_500 = clean_text_snippet(clean_text, 500)
    now_iso = datetime.utcnow().isoformat() + "Z"
    model_id = "antigravity-agent-v1"
    
    qa_list = []
    
    # Q&A 1: Core Subject / Mandate
    q1 = f"What is the primary subject and policy mandate notified under {doc_label} {source_ref_id}?"
    a1 = f"According to {doc_label} {source_ref_id} (dated {source_date}), the primary subject is: {subject}. Specifically, the document mandates: {summary_p1}"
    
    qa_list.append({
        "qa_id": f"qa_{doc_type.lower()}_{source_ref_id}_{idx_start}",
        "doc_type": doc_type,
        "source_ref_id": source_ref_id,
        "source_title": subject if len(subject) < 300 else subject[:300],
        "source_date": source_date,
        "question": q1,
        "answer": a1,
        "context_snippet": snippet_500,
        "generator_model": model_id,
        "generated_at": now_iso
    })
    
    # Q&A 2: Specific procedural/item conditions
    q2 = f"What specific procedural guidelines, schedule conditions, or item details are specified in {doc_label} {source_ref_id} regarding {topic}?"
    a2 = f"Under {source_ref_id}, the following specific conditions and guidelines apply: {summary_p2}. Exporters, importers, and relevant trade authorities are required to comply with these provisions."
    
    qa_list.append({
        "qa_id": f"qa_{doc_type.lower()}_{source_ref_id}_{idx_start + 1}",
        "doc_type": doc_type,
        "source_ref_id": source_ref_id,
        "source_title": subject if len(subject) < 300 else subject[:300],
        "source_date": source_date,
        "question": q2,
        "answer": a2,
        "context_snippet": snippet_500,
        "generator_model": model_id,
        "generated_at": now_iso
    })
    
    # Q&A 3: Authority / Effective Date (if text is substantial enough)
    if len(clean_text) > 700 and len(operative_sentences) >= 3:
        q3 = f"What is the effective timeline and legal authority underlying {doc_label} {source_ref_id}?"
        a3 = f"{doc_label} {source_ref_id} is issued under the authority of the Ministry of Commerce & Industry / Directorate General of Foreign Trade, with an official document date of {source_date}. The operational details and legal instructions outlined (`{topic}`) take effect as specified in the official notification provisions."
        
        qa_list.append({
            "qa_id": f"qa_{doc_type.lower()}_{source_ref_id}_{idx_start + 2}",
            "doc_type": doc_type,
            "source_ref_id": source_ref_id,
            "source_title": subject if len(subject) < 300 else subject[:300],
            "source_date": source_date,
            "question": q3,
            "answer": a3,
            "context_snippet": snippet_500,
            "generator_model": model_id,
            "generated_at": now_iso
        })
        
    return qa_list

def run_direct_generation():
    output_path = "data/processed/policy_qa_dataset.jsonl"
    
    # 1. Load existing rows and track covered (doc_type, source_ref_id)
    existing_count = 0
    covered_pairs = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for l in f:
                if not l.strip():
                    continue
                existing_count += 1
                row = json.loads(l)
                covered_pairs.add((row.get("doc_type"), str(row.get("source_ref_id"))))
                
    total_new_qa = 0
    unique_covered = {"DGFT_OCR": set(), "DGFT_POLICY": set(), "PIB_PRESS_RELEASE": set()}
    ocr_skipped_garbled = 0
    
    new_qa_rows = []
    
    # A. DGFT OCR Chunks (382 chunks)
    ocr_path = "data/processed/dgft_ocr_chunks.jsonl"
    if os.path.exists(ocr_path):
        with open(ocr_path, "r", encoding="utf-8") as f:
            for l in f:
                if not l.strip():
                    continue
                chunk = json.loads(l)
                ref_id = str(chunk.get("notification_no", chunk.get("chunk_id", "REF")))
                doc_type = "DGFT_OCR"
                
                if (doc_type, ref_id) in covered_pairs:
                    continue
                    
                clean_t = chunk.get("clean_text", "")
                if is_garbled_ocr(clean_t):
                    ocr_skipped_garbled += 1
                    continue
                    
                qas = generate_qa_for_chunk(chunk, doc_type, idx_start=1)
                new_qa_rows.extend(qas)
                unique_covered[doc_type].add(ref_id)
                covered_pairs.add((doc_type, ref_id))

    # B. DGFT Policy Chunks (check any uncovered by chunk_id/notification_no)
    dgft_path = "data/processed/dgft_policy_chunks.jsonl"
    if os.path.exists(dgft_path):
        with open(dgft_path, "r", encoding="utf-8") as f:
            for l in f:
                if not l.strip():
                    continue
                chunk = json.loads(l)
                ref_id = str(chunk.get("notification_no", chunk.get("chunk_id", "REF")))
                doc_type = "DGFT_POLICY"
                
                if (doc_type, ref_id) in covered_pairs and (doc_type, str(chunk.get("chunk_id"))) in covered_pairs:
                    continue
                    
                clean_t = chunk.get("clean_text", "")
                if len(clean_t.strip()) < 80:
                    continue
                    
                qas = generate_qa_for_chunk(chunk, doc_type, idx_start=1)
                new_qa_rows.extend(qas)
                unique_covered[doc_type].add(ref_id)
                covered_pairs.add((doc_type, ref_id))
                covered_pairs.add((doc_type, str(chunk.get("chunk_id"))))

    # C. PIB Press Releases (check uncovered)
    pib_path = "data/processed/pib_press_releases.jsonl"
    if os.path.exists(pib_path):
        with open(pib_path, "r", encoding="utf-8") as f:
            for l in f:
                if not l.strip():
                    continue
                chunk = json.loads(l)
                ref_id = str(chunk.get("prid", "PRID"))
                doc_type = "PIB_PRESS_RELEASE"
                
                if (doc_type, ref_id) in covered_pairs:
                    continue
                    
                clean_t = chunk.get("clean_text", "")
                if len(clean_t.strip()) < 80:
                    continue
                    
                qas = generate_qa_for_chunk(chunk, doc_type, idx_start=1)
                new_qa_rows.extend(qas)
                unique_covered[doc_type].add(ref_id)
                covered_pairs.add((doc_type, ref_id))

    # 3. Append only ('a' mode)
    with open(output_path, "a", encoding="utf-8") as f:
        for r in new_qa_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            total_new_qa += 1
            
    # 4. Final verification of line count
    new_total_lines = 0
    with open(output_path, "r", encoding="utf-8") as f:
        new_total_lines = sum(1 for _ in f if _.strip())
        
    print(f"Total new QA pairs added: {total_new_qa}")
    print(f"Unique docs covered per doc_type:")
    for dt, s in unique_covered.items():
        print(f"  {dt}: {len(s)} unique docs")
    print(f"Count of OCR chunks skipped due to garbled/unreliable text: {ocr_skipped_garbled}")
    print(f"New total line count of policy_qa_dataset.jsonl: {new_total_lines}")

if __name__ == "__main__":
    run_direct_generation()
