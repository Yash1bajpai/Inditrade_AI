"""
OCR Extractor for Scanned DGFT Policy Circulars and Notifications.

Option B Pipeline:
1. Preprocesses scanned PDF pages (Grayscale + Otsu Binarization via OpenCV).
2. Extracts text using Tesseract OCR.
3. Applies strict Quality Control (QC) Gate (Length check, Gibberish/Noise ratio check).
4. Safely appends clean chunks to `data/processed/dgft_ocr_chunks.jsonl` without modifying
   the original digital chunks (`dgft_policy_chunks.jsonl`).
"""

import os
import sys
import glob
import json
import re
import random
import string
import argparse
from typing import List, Dict, Tuple, Any

import cv2
import numpy as np
import pdf2image
import pytesseract

# Configure standard Windows paths for Tesseract and Poppler if not already set or in PATH
DEFAULT_TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
DEFAULT_POPPLER_BIN = r"C:\Users\Admin\AppData\Local\Microsoft\WinGet\Packages\oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe\poppler-25.07.0\Library\bin"

if os.path.exists(DEFAULT_TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESSERACT_CMD

POPPLER_PATH = DEFAULT_POPPLER_BIN if os.path.exists(DEFAULT_POPPLER_BIN) else None

# Ensure proper terminal encoding for special characters
sys_reconfig = getattr(sys.stdout, 'reconfigure', None)
if sys_reconfig:
    sys_reconfig(encoding='utf-8', errors='replace')


def preprocess_image_for_ocr(pil_image) -> np.ndarray:
    """
    Convert PIL Image to OpenCV Grayscale and apply Otsu Binarization (thresholding)
    to eliminate faint watermarks, stamps, and background noise.
    """
    # Convert PIL RGB to numpy BGR/RGB
    img_np = np.array(pil_image)
    
    # Convert to Grayscale
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
        
    # Apply Otsu's Thresholding (Binarization)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return thresh


def run_qc_gate(raw_text: str) -> Tuple[bool, str, str, float]:
    """
    Quality Control (QC) Gate:
    1. Clean up multiple newlines and spaces.
    2. Length Check: Drop if fewer than 100 characters.
    3. Gibberish Check: Calculate ratio of non-alphanumeric chars (excluding spaces/punctuation).
       If ratio > 15% (0.15) OR alphanumeric chars < 45%, DROP as stamp/signature/distorted table.
       
    Returns: (passed: bool, clean_text: str, drop_reason: str, noise_ratio: float)
    """
    # Clean up whitespace and newlines
    clean_text = re.sub(r'\s+', ' ', raw_text).strip()
    
    # Length Check
    if len(clean_text) < 100:
        return False, clean_text, f"Length ({len(clean_text)} chars) < 100 threshold", 0.0
        
    # Gibberish Check
    total_len = len(clean_text)
    alnum_count = sum(1 for c in clean_text if c.isalnum())
    
    # Non-alphanumeric excluding spaces and standard punctuation
    noise_chars = [
        c for c in clean_text 
        if not c.isalnum() and not c.isspace() and c not in string.punctuation
    ]
    noise_ratio = len(noise_chars) / total_len
    alnum_ratio = alnum_count / total_len
    
    if noise_ratio > 0.15:
        return False, clean_text, f"Gibberish/Noise ratio ({noise_ratio:.1%}) exceeds 15% limit", noise_ratio
        
    if alnum_ratio < 0.45:
        return False, clean_text, f"Alphanumeric density ({alnum_ratio:.1%}) too low (<45% - likely distorted table/stamp)", noise_ratio
        
    return True, clean_text, "PASSED", noise_ratio


def parse_metadata_from_filename(filename: str) -> Dict[str, str]:
    """Extract notification number, date estimate, and circular type from filename."""
    base = os.path.splitext(os.path.basename(filename))[0]
    
    # Detect type
    c_type = "Circular"
    if "Notification" in base:
        c_type = "Notification"
    elif "Trade_Notice" in base or "Trade Notice" in base:
        c_type = "Trade_Notice"
    elif "Public_Notice" in base or "Public Notice" in base:
        c_type = "Public_Notice"
        
    # Extract number (e.g. Notification_01_2023 -> 01/2023)
    parts = base.split('_')
    notif_no = "Unknown"
    for idx, p in enumerate(parts):
        if p.isdigit() and idx + 1 < len(parts) and (parts[idx+1].isdigit() or '-' in parts[idx+1]):
            notif_no = f"{p}/{parts[idx+1]}"
            break
        elif p.isdigit() and len(p) <= 3:
            notif_no = p
            break
            
    # Estimate date from filename numbers or default to 2024/2025
    date_str = "01/01/2024"
    if "2023" in base:
        date_str = "15/06/2023"
    elif "2024" in base:
        date_str = "15/06/2024"
    elif "2025" in base:
        date_str = "15/06/2025"
    elif "2026" in base:
        date_str = "15/06/2026"
        
    return {
        "notification_no": notif_no,
        "type": c_type,
        "date": date_str
    }


def find_scanned_pdfs(all_pdf_dir: str, processed_jsonl: str) -> List[str]:
    """Identify the remaining scanned PDFs that were not processed/converted in the digital phase."""
    all_pdfs = sorted(glob.glob(os.path.join(all_pdf_dir, "*.pdf")))
    
    processed_paths = set()
    if os.path.exists(processed_jsonl):
        with open(processed_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "pdf_path" in data:
                            processed_paths.add(os.path.normpath(data["pdf_path"]))
                    except Exception:
                        pass
                        
    scanned_pdfs = []
    for p in all_pdfs:
        if os.path.normpath(p) not in processed_paths:
            scanned_pdfs.append(p)
            
    return scanned_pdfs


def run_ocr_pipeline(sample_size: int = 5, seed: int = 42, output_file: str = "data/processed/dgft_ocr_chunks.jsonl"):
    print(f"=== IndiTrade AI: Option B OCR Extraction & QC Gate Pipeline ===")
    
    # 1. Identify scanned PDFs
    scanned_pdfs = find_scanned_pdfs(
        all_pdf_dir="data/raw/dgft_notifications/pdfs",
        processed_jsonl="data/processed/dgft_policy_chunks.jsonl"
    )
    print(f"Total remaining scanned PDFs identified: {len(scanned_pdfs)}")
    
    if len(scanned_pdfs) == 0:
        print("No scanned PDFs found to process.")
        return
        
    # 2. Select exact sample size
    random.seed(seed)
    selected_pdfs = random.sample(scanned_pdfs, min(sample_size, len(scanned_pdfs)))
    print(f"Randomly selected exactly {len(selected_pdfs)} scanned PDFs for OCR processing:\n")
    for idx, sp in enumerate(selected_pdfs, 1):
        print(f"  [{idx}] {os.path.basename(sp)}")
    print()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    total_pages_processed = 0
    passed_qc_count = 0
    dropped_qc_count = 0
    passed_sample_chunks = []
    
    # 3. Process selected PDFs
    for pdf_idx, pdf_path in enumerate(selected_pdfs, 1):
        meta = parse_metadata_from_filename(pdf_path)
        print(f"\n[{pdf_idx}/{len(selected_pdfs)}] Processing: {os.path.basename(pdf_path)} (Notif: {meta['notification_no']})")
        
        try:
            # Convert PDF pages to PIL images
            images = pdf2image.convert_from_path(
                pdf_path,
                poppler_path=POPPLER_PATH,
                dpi=300
            )
            print(f"  -> Converted {len(images)} page(s) to 300 DPI images.")
            
            # Combine pages or process page-by-page. For high-quality RAG, we OCR page by page
            # and consolidate into a single clean chunk if quality allows, or individual page chunks.
            full_clean_text = []
            page_qc_passed = 0
            
            for page_num, pil_img in enumerate(images, 1):
                total_pages_processed += 1
                # Preprocessing: Grayscale + Otsu Binarization
                processed_img = preprocess_image_for_ocr(pil_img)
                
                # Tesseract OCR extraction
                page_raw_text = pytesseract.image_to_string(processed_img, lang="eng")
                
                # Run QC Gate
                passed, page_clean_text, reason, noise_ratio = run_qc_gate(page_raw_text)
                
                if passed:
                    page_qc_passed += 1
                    full_clean_text.append(f"[Page {page_num}]\n{page_clean_text}")
                    print(f"     Page {page_num}: ✅ QC PASSED (Length: {len(page_clean_text)} chars, Noise: {noise_ratio:.1%})")
                else:
                    print(f"     Page {page_num}: ❌ QC DROPPED -> Reason: {reason}")
                    
            if page_qc_passed > 0 and len(full_clean_text) > 0:
                consolidated_text = "\n\n".join(full_clean_text)
                # Final check on consolidated chunk
                passed_cons, cons_clean, cons_reason, cons_noise = run_qc_gate(consolidated_text)
                if passed_cons:
                    passed_qc_count += 1
                    chunk_id = f"DGFT_OCR_{meta['type'].upper()}_{meta['notification_no'].replace('/', '_')}_{pdf_idx}"
                    chunk_data = {
                        "chunk_id": chunk_id,
                        "notification_no": meta["notification_no"],
                        "type": meta["type"],
                        "date": meta["date"],
                        "pdf_path": pdf_path,
                        "clean_text": cons_clean,
                        "qc_status": "PASSED",
                        "ocr_pages_passed": page_qc_passed,
                        "ocr_total_pages": len(images)
                    }
                    
                    # Safe Append
                    with open(output_file, "a", encoding="utf-8") as out_f:
                        out_f.write(json.dumps(chunk_data, ensure_ascii=False) + "\n")
                        
                    if len(passed_sample_chunks) < 2:
                        passed_sample_chunks.append(chunk_data)
                else:
                    dropped_qc_count += 1
                    print(f"  -> Consolidated chunk dropped: {cons_reason}")
            else:
                dropped_qc_count += 1
                print(f"  -> Entire PDF dropped (0 pages passed QC Gate).")
                
        except Exception as e:
            dropped_qc_count += 1
            print(f"  -> ERROR processing PDF: {str(e)}")
            
    # 4. Print Execution Summary Logs
    print("\n" + "="*60)
    print("=== OCR EXTRACTION & QUALITY CONTROL (QC) GATE SUMMARY ===")
    print("="*60)
    print(f"Total Scanned PDFs Selected  : {len(selected_pdfs)}")
    print(f"Total PDF Pages Processed    : {total_pages_processed}")
    print(f"Chunks PASSED QC Gate (Saved): {passed_qc_count}")
    print(f"Chunks DROPPED as Garbage    : {dropped_qc_count}")
    print(f"Target Output File           : {output_file}")
    print("="*60 + "\n")
    
    # 5. Print 2 Actual Chunks for Manual Verification
    if passed_sample_chunks:
        print("=== ACTUAL CHUNKS THAT PASSED QC GATE (MANUAL VERIFICATION) ===\n")
        for idx, chunk in enumerate(passed_sample_chunks, 1):
            print(f"--- SAMPLE CHUNK #{idx} ---")
            print(f"Chunk ID         : {chunk['chunk_id']}")
            print(f"Notification No  : {chunk['notification_no']}")
            print(f"Type / Date      : {chunk['type']} ({chunk['date']})")
            print(f"Pages OCR'd      : {chunk['ocr_pages_passed']} / {chunk['ocr_total_pages']}")
            print(f"Text Length      : {len(chunk['clean_text'])} characters")
            print("Extracted Clean Text Snippet:")
            print("-" * 50)
            print(chunk["clean_text"][:800].replace('\r', ' '))
            print("-" * 50 + "\n")
    else:
        print("No chunks passed the QC Gate across the selected sample.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OCR Extraction with QC Gate on Scanned DGFT PDFs")
    parser.add_argument("--sample", type=int, default=5, help="Number of randomly selected PDFs to process (Default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection (Default: 42)")
    args = parser.parse_args()
    
    run_ocr_pipeline(sample_size=args.sample, seed=args.seed)
