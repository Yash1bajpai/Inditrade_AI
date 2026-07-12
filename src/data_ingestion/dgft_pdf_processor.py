"""
IndiTrade AI - DGFT Master Scraper, PDF Downloader & Text Chunk Processor
1. Scrapes full list of real DGFT Notifications, Public Notices, and Trade Notices (500+ records).
2. Downloads actual official PDF documents into data/raw/dgft_notifications/pdfs/.
3. Extracts and cleans text using pdfplumber (stripping headers, footers, and noise).
4. Saves clean policy chunks to data/processed/dgft_policy_chunks.jsonl.
"""

import os
import re
import sys
import json
import time
import requests
import pandas as pd
import pdfplumber
from datetime import datetime

# Set safe UTF-8 output encoding for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Directories
RAW_DIR = os.path.join("data", "raw", "dgft_notifications")
PDF_DIR = os.path.join(RAW_DIR, "pdfs")
PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_JSONL = os.path.join(PROCESSED_DIR, "dgft_policy_chunks.jsonl")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Connection": "keep-alive"
}

def clean_filename(notif_no, subject, type_str):
    """Generates a clean local filename from notification number and subject."""
    clean_no = re.sub(r'[^a-zA-Z0-9_-]', '_', str(notif_no)).strip('_')
    clean_subj = re.sub(r'[^a-zA-Z0-9_-]', '_', str(subject)[:40]).strip('_')
    return f"{type_str}_{clean_no}_{clean_subj}.pdf"

def scrape_full_dgft_master_list():
    """
    Scrapes all rows from Notification, Public Notice, and Trade Notice endpoints.
    Correctly maps table columns:
    cols[1] = Notification No., cols[2] = Financial Year, cols[3] = Subject, cols[4] = Issue Date.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    print("=== [STEP 1] SCRAPING MASTER LIST OF DGFT NOTIFICATIONS/NOTICES ===")
    
    targets = [
        {"Type": "Notification", "Url": "https://www.dgft.gov.in/CP/?opt=notification"},
        {"Type": "Public_Notice", "Url": "https://www.dgft.gov.in/CP/?opt=public-notice"},
        {"Type": "Trade_Notice", "Url": "https://www.dgft.gov.in/CP/?opt=trade-notice"}
    ]
    
    all_records = []
    
    for t in targets:
        print(f"[*] Fetching table from {t['Url']}...", end=" ", flush=True)
        try:
            resp = requests.get(t["Url"], headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                print(f"[FAILED] HTTP {resp.status_code}")
                continue
                
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.content, "html.parser")
            rows = soup.find_all("tr")
            
            count = 0
            for row in rows:
                cols = row.find_all("td")
                if len(cols) >= 5:
                    notif_no = cols[1].text.strip()
                    fin_year = cols[2].text.strip()
                    subject = cols[3].text.strip()
                    issue_date = cols[4].text.strip()
                    
                    pdf_link = ""
                    a_tag = row.find("a", href=True)
                    if a_tag:
                        href = a_tag["href"]
                        if href.startswith("http"):
                            pdf_link = href
                        elif href.startswith("/"):
                            pdf_link = f"https://www.dgft.gov.in{href}"
                        else:
                            pdf_link = f"https://www.dgft.gov.in/CP/{href}"
                            
                    if notif_no and subject and pdf_link:
                        local_pdf_name = clean_filename(notif_no, subject, t["Type"])
                        local_pdf_path = os.path.join(PDF_DIR, local_pdf_name)
                        
                        all_records.append({
                            "Type": t["Type"],
                            "Notification_No": notif_no,
                            "Financial_Year": fin_year,
                            "Subject": subject,
                            "Issue_Date": issue_date,
                            "PDF_Link": pdf_link,
                            "Local_PDF_Path": local_pdf_path
                        })
                        count += 1
                        
            print(f"[SUCCESS] Extracted {count} records.")
            time.sleep(1.0)
            
        except Exception as e:
            print(f"[ERROR] Exception scraping {t['Type']}: {e}")
            
    df_master = pd.DataFrame(all_records)
    if not df_master.empty:
        df_master = df_master.drop_duplicates(subset=["Type", "Notification_No", "Subject"]).reset_index(drop=True)
        master_csv = os.path.join(RAW_DIR, "dgft_master_list.csv")
        df_master.to_csv(master_csv, index=False)
        print(f"\n[MASTER LIST COMPLETE] Total Unique DGFT Circulars/Notifications Found: {len(df_master)}")
        print(f"Saved Master CSV: {master_csv}")
    else:
        print("\n[WARNING] No records found.")
        
    return df_master

def download_pdfs(df_master, limit=None):
    """
    Downloads actual PDF documents into data/raw/dgft_notifications/pdfs/.
    Skips if file already exists and has valid size (>500 bytes).
    """
    print(f"\n=== [STEP 2] DOWNLOADING OFFICIAL DGFT PDF DOCUMENTS ===")
    
    if limit:
        df_master = df_master.head(limit)
        
    downloaded_count = 0
    existing_count = 0
    failed_count = 0
    
    total = len(df_master)
    
    for idx, row in df_master.iterrows():
        path = row["Local_PDF_Path"]
        url = row["PDF_Link"]
        notif_no = str(row["Notification_No"]).encode('ascii', 'replace').decode('ascii')
        
        if os.path.exists(path) and os.path.getsize(path) > 500:
            existing_count += 1
            continue
            
        print(f"[{idx+1:03d}/{total}] Downloading {row['Type']} {notif_no[:30]}...", end=" ", flush=True)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=12)
            if resp.status_code == 200 and len(resp.content) > 100:
                with open(path, "wb") as f:
                    f.write(resp.content)
                downloaded_count += 1
                print(f"[SUCCESS] {os.path.getsize(path)//1024} KB")
            else:
                failed_count += 1
                print(f"[FAILED] HTTP {resp.status_code}")
        except Exception as e:
            failed_count += 1
            err_msg = str(e).encode('ascii', 'replace').decode('ascii')
            print(f"[ERROR] {err_msg[:30]}")
            
        time.sleep(0.25)
        
    print(f"\n[PDF DOWNLOAD SUMMARY] Newly Downloaded: {downloaded_count} | Already Existing: {existing_count} | Failed/Broken Link: {failed_count}")
    return downloaded_count + existing_count

def clean_pdf_text(raw_text):
    """
    Cleans raw PDF text extracted by pdfplumber:
    - Removes headers/footers (`TO BE PUBLISHED IN THE GAZETTE...`, `DIRECTORATE GENERAL...`, page numbers).
    - Removes extra whitespace and non-ASCII noise while preserving clean paragraphs.
    """
    if not raw_text:
        return ""
        
    lines = raw_text.splitlines()
    clean_lines = []
    
    # Common header/footer/noise patterns
    noise_patterns = [
        r'TO BE PUBLISHED IN THE GAZETTE OF INDIA',
        r'EXTRAORDINARY, PART(-| )II',
        r'SECTION (-| )3, SUB(-| )SECTION',
        r'GOVERNMENT OF INDIA',
        r'MINISTRY OF COMMERCE AND INDUSTRY',
        r'DEPARTMENT OF COMMERCE',
        r'DIRECTORATE GENERAL OF FOREIGN TRADE',
        r'VANIJYA BHAWAN, NEW DELHI',
        r'^\s*Page \d+ of \d+\s*$',
        r'^\s*-+\s*$',
        r'^\s*\d+\s*$'
    ]
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
            
        is_noise = False
        for pat in noise_patterns:
            if re.search(pat, stripped, re.IGNORECASE):
                is_noise = True
                break
                
        if not is_noise:
            clean_lines.append(stripped)
            
    # Join and normalize multiple spaces
    text = "\n".join(clean_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def process_pdfs_to_chunks(df_master):
    """
    Extracts text from all downloaded PDFs using pdfplumber, cleans text, and saves JSONL.
    Output: data/processed/dgft_policy_chunks.jsonl
    """
    print(f"\n=== [STEP 3] EXTRACTING & CLEANING TEXT USING PDFPLUMBER ===")
    
    chunks = []
    processed_count = 0
    total = len(df_master)
    
    for idx, row in df_master.iterrows():
        path = row["Local_PDF_Path"]
        if not os.path.exists(path) or os.path.getsize(path) < 500:
            continue
            
        try:
            full_raw_text = ""
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    txt = page.extract_text()
                    if txt:
                        full_raw_text += txt + "\n"
                        
            clean_text = clean_pdf_text(full_raw_text)
            
            # If text extraction via pdfplumber got clean content
            if clean_text and len(clean_text) > 30:
                chunk_id = f"DGFT_{row['Type'].upper()}_{re.sub(r'[^a-zA-Z0-9]', '_', str(row['Notification_No']))}"
                chunk_record = {
                    "chunk_id": chunk_id,
                    "notification_no": str(row["Notification_No"]),
                    "type": str(row["Type"]),
                    "date": str(row["Issue_Date"]),
                    "financial_year": str(row["Financial_Year"]),
                    "subject": str(row["Subject"]),
                    "pdf_path": str(path),
                    "clean_text": clean_text
                }
                chunks.append(chunk_record)
                processed_count += 1
                if processed_count % 50 == 0:
                    print(f"[*] Processed {processed_count} PDF chunks...")
        except Exception as e:
            pass
            
    # Write to JSONL
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            
    print(f"\n[PROCESSING COMPLETE] Total Clean Policy Chunks Created: {len(chunks)}")
    print(f"Saved Output JSONL: {OUTPUT_JSONL}")
    return chunks

if __name__ == "__main__":
    df_master = scrape_full_dgft_master_list()
    if not df_master.empty:
        download_pdfs(df_master)
        chunks = process_pdfs_to_chunks(df_master)
