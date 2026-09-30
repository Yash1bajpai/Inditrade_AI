"""
IndiTrade AI - PIB Press Release Scraper for Trade, Tariff & Macro Policy
Scrapes real official press releases from pib.gov.in using BeautifulSoup4.
Extracts full article body text, ministry, date, and policy classification.
Output: data/processed/pib_press_releases.jsonl
"""

import os
import re
import sys
import json
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_JSONL = os.path.join(PROCESSED_DIR, "pib_press_releases.jsonl")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,hi;q=0.8",
    "Connection": "keep-alive"
}

POLICY_KEYWORDS = [
    'trade', 'tariff', 'export', 'import', 'commerce', 'finance', 'economy', 'tax', 'gst', 'customs',
    'pli', 'gdp', 'investment', 'industry', 'policy', 'fdi', 'duty', 'exim', 'roftep', 'rodtep', 'dgft',
    'subsidy', 'banking', 'rbi', 'budget', 'fiscal', 'inflation', 'forex', 'rupee', 'manufacturing',
    'व्यापार', 'निर्यात', 'आयात', 'वाणिज्य', 'वित्त', 'अर्थव्यवस्था', 'कर', 'शुल्क', 'नीति', 'उद्योग',
    'निवेश', 'सीमा शुल्क', 'जीएसटी', 'राजकोषीय', 'मुद्रास्फीति', 'बजट', 'विनिर्माण'
]

def clean_article_text(raw_text):
    """
    Cleans raw PIB iframe HTML text:
    - Removes JavaScript blocks, navigation footers (`Click here for Release...`), and boilerplate.
    - Normalizes spacing and paragraphs.
    """
    if not raw_text:
        return ""
    lines = raw_text.splitlines()
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if 'JavaScript must be enabled' in stripped or 'document.write' in stripped or 'Click here for Release' in stripped:
            continue
        if stripped.startswith('***') or stripped.startswith('(रिलीज़ आईड'):
            continue
        clean_lines.append(stripped)
    text = "\n".join(clean_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def classify_policy_category(title, text):
    """Assigns high-level policy category based on keywords."""
    combined = (title + " " + text).lower()
    if any(k in combined for k in ['trade', 'tariff', 'export', 'import', 'customs', 'duty', 'dgft', 'exim', 'व्यापार', 'निर्यात', 'आयात', 'सीमा शुल्क']):
        return "Trade & Tariff Policy"
    elif any(k in combined for k in ['finance', 'tax', 'gst', 'budget', 'fiscal', 'rbi', 'banking', 'inflation', 'rupee', 'forex', 'वित्त', 'कर', 'जीएसटी', 'बजट']):
        return "Macro & Finance Policy"
    else:
        return "Industrial & Economic Policy"

def collect_candidate_prids():
    """
    Scrapes listing endpoints on pib.gov.in to collect all active PRID links and titles.
    Filters for relevance to Trade, Tariff, Export, Commerce, and Macro Economics.
    """
    print("=== [STEP 1] COLLECTING REAL PRESS RELEASE CANDIDATES FROM PIB.GOV.IN ===")

    listing_urls = [
        "https://pib.gov.in/AllRelease.aspx?MenuId=3&Lang=1",
        "https://pib.gov.in/AllRelease.aspx?MenuId=3&Lang=2",
        "https://pib.gov.in/AllRelease.aspx?MinId=15",
        "https://pib.gov.in/AllRelease.aspx?MinId=4",
        "https://pib.gov.in/indexd.aspx"
    ]

    candidates = {}

    for url in listing_urls:
        print(f"[*] Checking listing page: {url}...", end=" ", flush=True)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=12)
            if resp.status_code != 200:
                print(f"[FAILED] HTTP {resp.status_code}")
                continue

            soup = BeautifulSoup(resp.content, "html.parser")
            links = soup.find_all("a", href=True)

            count = 0
            for a in links:
                href = a["href"]
                if "PRID=" in href:
                    prid = href.split("PRID=")[-1].split("&")[0].strip()
                    if not prid.isdigit():
                        continue
                    title = a.text.strip()
                    if not title or len(title) < 10:
                        parent = a.find_parent(["li", "tr", "div"])
                        if parent:
                            title = parent.text.strip()[:150].replace('\n', ' ')

                    is_relevant = any(k.lower() in title.lower() for k in POLICY_KEYWORDS)
                    if False: # Never add unrelated articles to meet a quota.

                        is_relevant = True

                    if is_relevant and prid not in candidates:
                        candidates[prid] = {
                            "prid": prid,
                            "title": title,
                            "source_url": url
                        }
                        count += 1

            print(f"[SUCCESS] Added {count} new candidates.")
            time.sleep(1.0)

        except Exception as e:
            err_msg = str(e).encode('ascii', 'replace').decode('ascii')
            print(f"[ERROR] {err_msg[:40]}")

    print(f"\n[CANDIDATE COLLECTION COMPLETE] Total Unique Candidate PRIDs Collected: {len(candidates)}")
    return list(candidates.values())

def scrape_full_articles(candidates, max_articles=70):
    """
    Scrapes exact full article text for collected PRIDs from PressReleaseIframePage.aspx.
    Strictly adheres to custom User-Agent + 1s delay.
    """
    print(f"\n=== [STEP 2] SCRAPING FULL ARTICLE TEXT FOR UP TO {max_articles} CANDIDATES ===")

    articles = []
    total = min(len(candidates), max_articles)

    for idx, cand in enumerate(candidates[:max_articles]):
        prid = cand["prid"]

        iframe_urls = [
            f"https://pib.gov.in/PressReleaseIframePage.aspx?PRID={prid}&RegId=3&Lang=1",
            f"https://pib.gov.in/PressReleaseIframePage.aspx?PRID={prid}&RegId=3&Lang=2",
            f"https://pib.gov.in/PressReleaseIframePage.aspx?PRID={prid}"
        ]

        full_text = ""
        ministry = "Press Information Bureau / Government of India"
        date_str = ""
        clean_title = cand["title"]

        print(f"[{idx+1:03d}/{total}] Scraping PRID {prid}...", end=" ", flush=True)

        success = False
        for i_url in iframe_urls:
            try:
                resp = requests.get(i_url, headers=HEADERS, timeout=12)
                if resp.status_code == 200 and len(resp.content) > 500:
                    soup = BeautifulSoup(resp.content, "html.parser")
                    date_node = soup.select_one(".ReleaseDateSubHeaddateTime")
                    date_match = re.search(r"(\d{1,2}\s+[A-Z]{3}\s+\d{4})", date_node.get_text(" ", strip=True) if date_node else "", re.I)
                    if not date_match:
                        continue
                    date_str = datetime.strptime(date_match.group(1).upper(), "%d %b %Y").strftime("%Y-%m-%d")
                    body = soup.select_one(".innner-page-main-about-us-content-right-part") or soup
                    for unwanted in body.select("script, style"):
                        unwanted.decompose()
                    raw_text = body.get_text("\n", strip=True)
                    cleaned = clean_article_text(raw_text)

                    if len(cleaned) > 200:
                        full_text = cleaned

                        lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
                        if len(lines) >= 2:
                            if any(w in lines[0] for w in ['Ministry', 'Prime Minister', 'Secretariat', 'मंत्रालय', 'सचिवालय']):
                                ministry = lines[0]
                            elif any(w in lines[1] for w in ['Ministry', 'Prime Minister', 'Secretariat', 'मंत्रालय', 'सचिवालय']):
                                ministry = lines[1]
                        success = True
                        break
            except Exception:
                pass
            time.sleep(0.3)

        if success and full_text:
            category = classify_policy_category(clean_title, full_text)
            article_record = {
                "prid": str(prid),
                "title": clean_title,
                "ministry": ministry,
                "date": date_str,
                "url": f"https://pib.gov.in/PressReleasePage.aspx?PRID={prid}",
                "category": category,
                "clean_text": full_text,
                "is_fallback": False
            }
            articles.append(article_record)
            print(f"[SUCCESS] Extracted {len(full_text)} characters ({category})")
        else:
            print("[FAILED] Could not extract sufficient body text.")

        time.sleep(1.0)

    return articles

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    candidates = collect_candidate_prids()
    articles = scrape_full_articles(candidates, max_articles=65)

    if not articles or len(articles) == 0:
        raise RuntimeError("Live PIB scraping returned no valid articles; preserving previous corpus")

    # Merge into historical corpus rather than silently dropping older policies.
    existing = {}
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if not row.get("is_fallback"):
                        existing[row["prid"]] = row
    existing.update({row["prid"]: row for row in articles})
    articles = list(existing.values())
    tmp_path = OUTPUT_JSONL + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for item in articles:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, OUTPUT_JSONL)

    print(f"\n[SCRAPING & PROCESSING COMPLETE] Total Articles Saved: {len(articles)}")
    print(f"Saved Output JSONL: {OUTPUT_JSONL}")
    return articles

if __name__ == "__main__":
    main()
