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
# Verification stays ON. Preferred: `truststore` (standard, uses the OS store on
# Windows/macOS/Linux). Fallback: the Debian/Ubuntu system bundle. Last resort:
# requests' default certifi bundle, which fails closed on this host.
try:
    import truststore
    truststore.inject_into_ssl()
    VERIFY = True
except ImportError:
    SYSTEM_CA = "/etc/ssl/certs/ca-certificates.crt"
    VERIFY = SYSTEM_CA if os.path.exists(SYSTEM_CA) else True
OUT_DIR = Path("data/raw/apeda_mirror")
PDF_DIR = OUT_DIR / "pdfs"
OUTPUT_JSONL = Path("data/processed/apeda_mirror_chunks.jsonl")  # text-layer PDFs only
QUARANTINE_JSONL = Path("data/processed/apeda_mirror_quarantine_ocr.jsonl")  # scans: NOT for answers
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
    """Return (number, date) keys of documents already in the corpus."""
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


def existing_titles():
    """Lower-cased titles in the committed index; used to catch same document, different number."""
    index = Path("data/cache/bm25_index.pkl.lzma")
    if not index.exists():
        return []
    import lzma
    import pickle
    with lzma.open(index, "rb") as f:
        return [(d.get("title") or "").lower().strip() for d in pickle.load(f)["docs"]]


def same_title(subject, titles, threshold=0.95):
    import difflib
    s = subject.lower().strip()[:80]
    return any(t and difflib.SequenceMatcher(None, s, t[:80]).ratio() >= threshold for t in titles)


def missing_rows(rows, have, titles=()):
    """Rows not in the corpus by number+date AND without a near-identical title.
    Still a heuristic: a row kept here is a candidate gap, not proof of absence."""
    return [r for r in rows
            if (norm_no(r["notification_no"]), r["date"]) not in have
            and not same_title(r["subject"], titles)]


def pdf_text(path, with_flag=False):
    import pdfplumber
    out = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                out.append(t)
    text = "\n".join(out).strip()
    ocr = False
    if len(text) < 100:  # scanned PDF: same OCR route the DGFT pipeline uses
        text, ocr = ocr_pdf(path), True
    return (text, ocr) if with_flag else text


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


def build_record(row, text, ocr=False):
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
        "ocr_used": ocr,
        "ocr_warning": ("OCR text: numbers, dates and clause wording can be wrong (seen: notification "
                        "number and date garbled). Use the mirror-table fields and the PDF as truth.") if ocr else "",
    }


def run(fy_filter="2026-27", limit=None):
    have = existing_keys()
    titles = existing_titles()
    todo = []
    for doc_type, url in SOURCES.items():
        resp = requests.get(url, headers=HEADERS, timeout=60, verify=VERIFY)
        resp.raise_for_status()
        rows = [r for r in parse_listing(resp.text, doc_type) if fy_filter in r["notification_no"]]
        gap = missing_rows(rows, have, titles)
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
            text, ocr = pdf_text(path, with_flag=True)
            if len(text) < 100:
                raise RuntimeError("no extractable text even after OCR; left out")
            records.append(build_record(row, text, ocr))
        except Exception as e:
            failed.append((row["notification_no"], row["type"], str(e)[:80]))
        time.sleep(0.5)
    if not records:
        raise RuntimeError("No APEDA mirror documents extracted; nothing written")
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    eligible, quarantined = split_by_ocr(records)
    for path, recs in ((OUTPUT_JSONL, eligible), (QUARANTINE_JSONL, quarantined)):
        tmp = str(path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    print(f"{len(eligible)} text-layer records -> {OUTPUT_JSONL}; {len(quarantined)} OCR scans quarantined "
          f"-> {QUARANTINE_JSONL}; skipped {len(failed)}: {failed}")
    return eligible, failed


def split_by_ocr(records):
    """OCR text garbles numbers and dates (seen on Notification 41/2026-27), so scans are
    quarantined until a person checks them against the page image."""
    return [r for r in records if not r["ocr_used"]], [r for r in records if r["ocr_used"]]


if __name__ == "__main__":
    run()
