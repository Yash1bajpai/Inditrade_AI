"""
IndiTrade AI - APEDA mirror of official DGFT notifications and trade notices.

DGFT's own website blocks automated access (HTTP 403), so this module reads the
public APEDA (Ministry of Commerce) mirror pages instead:
  https://apeda.gov.in/dgft-notifications
  https://apeda.gov.in/dgft-trade-notice

Limits (kept on every record, never hidden):
- The mirror lags DGFT by roughly 2-3 weeks (publish date vs "uploaded date").
- The mirror has no Public Notices section, so Public Notices are NOT covered.
- Records carry the mirror URL, DGFT publish date and mirror upload date.
- This module only writes a separate candidate file. It never edits the
  existing DGFT corpus and never promotes anything to production.
- It fails closed: a page that cannot be parsed raises, nothing is invented.
"""
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BASE = "https://apeda.gov.in"
SOURCES = {
    "Notification": f"{BASE}/dgft-notifications",
    "Trade_Notice": f"{BASE}/dgft-trade-notice",
}
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; IndiTradeBot/1.0)", "Accept": "*/*"}
# APEDA's emSign root is in the OS trust store but not in certifi's bundle.
# Verification stays ON; we only point requests at the system bundle.
SYSTEM_CA = "/etc/ssl/certs/ca-certificates.crt"
VERIFY = SYSTEM_CA if os.path.exists(SYSTEM_CA) else True
OUT_DIR = Path("data/raw/apeda_mirror")
PDF_DIR = OUT_DIR / "pdfs"
OUTPUT_JSONL = Path("data/processed/apeda_mirror_chunks.jsonl")
EXISTING_JSONL = Path("data/processed/dgft_policy_chunks.jsonl")
DISCLAIMER = ("Copied from the APEDA mirror of DGFT documents, which can lag DGFT by weeks "
              "and has no Public Notices. Check dgft.gov.in for the latest.")


def _text(node):
    return re.sub(r"\s+", " ", re.sub(r"<!--.*?-->", "", node.decode_contents(), flags=re.S))


def _plain(html):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html)).strip()


def _date(value):
    """DD/MM/YYYY, or "" when the mirror leaves the date blank (older rows)."""
    try:
        return datetime.strptime(value.strip(), "%d %B, %Y").strftime("%d/%m/%Y")
    except ValueError:
        return ""


def parse_listing(html, doc_type):
    """Return rows from one mirror table. Raises if the table is missing or empty."""
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue
        number, published, uploaded, subject = (_plain(_text(td)) for td in tds[:4])
        link = tr.find("a", href=True)
        if not (number and subject and link and link["href"].lower().endswith(".pdf")):
            continue
        if not _date(published):  # an undated record has no provenance; leave it out
            continue
        if any(r["notification_no"] == number and r["date"] == _date(published) and r["subject"] == subject for r in rows):
            continue  # the mirror lists some rows twice
        rows.append({
            "type": doc_type,
            "notification_no": number,
            "date": _date(published),
            "uploaded_date": _date(uploaded),
            "subject": subject,
            "pdf_url": urljoin(BASE, link["href"]),
        })
    if not rows:
        raise RuntimeError(f"APEDA {doc_type} listing empty or changed; refusing to continue")
    return rows


def norm_no(value):
    value = re.sub(r"\s", "", str(value)).replace("2027", "27")
    return re.sub(r"/2026-2027$", "/2026-27", value)


def existing_keys(path=EXISTING_JSONL):
    keys = set()
    if Path(path).exists():
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                keys.add((norm_no(r.get("notification_no", "")), r.get("date", "")))
    index = Path("data/cache/bm25_index.pkl.lzma")
    if not keys and index.exists():  # committed production index when the jsonl is not restored
        import lzma
        import pickle
        with lzma.open(index, "rb") as f:
            for d in pickle.load(f)["docs"]:
                if d.get("notification_no"):
                    keys.add((norm_no(d["notification_no"]), d.get("date", "")))
    return keys


def missing_rows(rows, have):
    return [r for r in rows if (norm_no(r["notification_no"]), r["date"]) not in have]


def pdf_text(path):
    import pdfplumber
    out = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                out.append(t)
    text = "\n".join(out).strip()
    if len(text) < 100:  # scanned PDF: same OCR route the DGFT pipeline uses
        text = ocr_pdf(path)
    return text


def ocr_pdf(path):
    import shutil
    import subprocess
    import tempfile
    if not (shutil.which("tesseract") and shutil.which("pdftoppm")):
        return ""
    parts = []
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-r", "150", "-png", str(path), os.path.join(tmp, "page")],
                       check=True, timeout=120, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for image in sorted(Path(tmp).glob("page-*.png")):
            r = subprocess.run(["tesseract", str(image), "stdout", "-l", "eng"],
                               check=True, timeout=60, capture_output=True, text=True)
            parts.append(r.stdout)
    return "\n".join(parts).strip()


def build_record(row, text):
    slug = re.sub(r"[^a-zA-Z0-9]", "_", row["notification_no"])
    return {
        "chunk_id": f"APEDA_{row['type'].upper()}_{slug}",
        "notification_no": row["notification_no"],
        "type": row["type"],
        "date": row["date"],
        "financial_year": "2026-27",
        "subject": row["subject"],
        "pdf_path": row["pdf_url"],
        "url": row["pdf_url"],
        "clean_text": text,
        "source": "APEDA_MIRROR_OF_DGFT",
        "mirror_uploaded_date": row["uploaded_date"],
        "source_note": DISCLAIMER,
    }


def run(fy_filter="2026-27", limit=None):
    have = existing_keys()
    todo = []
    for doc_type, url in SOURCES.items():
        resp = requests.get(url, headers=HEADERS, timeout=60, verify=VERIFY)
        resp.raise_for_status()
        rows = [r for r in parse_listing(resp.text, doc_type) if fy_filter in r["notification_no"]]
        gap = missing_rows(rows, have)
        print(f"{doc_type}: {len(rows)} mirror rows for {fy_filter}, {len(gap)} not in corpus")
        todo += gap
    if limit:
        todo = todo[:limit]
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    records, failed = [], []
    for row in todo:
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{row['type']}_{row['notification_no']}") + ".pdf"
        path = PDF_DIR / name
        try:
            r = requests.get(row["pdf_url"], headers=HEADERS, timeout=60, verify=VERIFY)
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"):
                raise RuntimeError("not a PDF")
            path.write_bytes(r.content)
            text = pdf_text(path)
            if len(text) < 100:
                raise RuntimeError("no extractable text even after OCR; left out")
            records.append(build_record(row, text))
        except Exception as e:
            failed.append((row["notification_no"], row["type"], str(e)[:80]))
        time.sleep(0.5)
    if not records:
        raise RuntimeError("No APEDA mirror documents extracted; nothing written")
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(OUTPUT_JSONL) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, OUTPUT_JSONL)
    print(f"Wrote {len(records)} records to {OUTPUT_JSONL}; skipped {len(failed)}: {failed}")
    return records, failed


if __name__ == "__main__":
    run()
