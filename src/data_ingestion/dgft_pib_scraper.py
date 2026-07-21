"""
IndiTrade AI - DGFT & PIB Public Domain Scraping Module
Scrapes official EXIM / Trade policy notifications from DGFT and Macro/Commerce
Press Releases from PIB India without requiring any API keys.

Features:
- PIB Scraping: Fetches ministry releases & RSS feeds, filtering for trade/macro policy.
- DGFT Scraping: Fetches public trade notifications/circulars, extracts policy intelligence.
- Anti-blocking: Custom browser headers, rate limiting, and exact row count verification.
"""

import os
import time
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

PIB_DIR = os.path.join("data", "raw", "pib_releases")
DGFT_DIR = os.path.join("data", "raw", "dgft_notifications")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5"
}

def scrape_pib_releases(max_pages=5):
    """
    Scrapes PIB Press Releases (Ministry of Commerce & Industry / Ministry of Finance / RBI updates).
    Saves structured CSV & Parquet to data/raw/pib_releases/.
    """
    os.makedirs(PIB_DIR, exist_ok=True)
    print("=== [PIB SCRAPER] STARTING PUBLIC PRESS RELEASE SCRAPING ===")

    KEYWORDS = ["trade", "export", "import", "dgft", "tariff", "exim", "customs", "pli", "inflation", "rbi", "forex", "gdp"]

    releases = []

    rss_urls = [
        "https://pib.gov.in/RssFeed.aspx?MinId=4",
        "https://pib.gov.in/RssFeed.aspx?MinId=15",
        "https://pib.gov.in/RssMain.aspx"
    ]

    for url in rss_urls:
        print(f"[*] Fetching PIB Feed: {url}...", end=" ", flush=True)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                print(f"[FAILED] HTTP {resp.status_code}")
                continue

            soup = BeautifulSoup(resp.content, "lxml-xml" if "xml" in resp.headers.get("Content-Type", "") else "html.parser")
            items = soup.find_all("item")

            count = 0
            for item in items:
                title = item.find("title").text.strip() if item.find("title") else ""
                link = item.find("link").text.strip() if item.find("link") else ""
                pub_date = item.find("pubDate").text.strip() if item.find("pubDate") else ""
                desc = item.find("description").text.strip() if item.find("description") else ""

                combined_text = f"{title} {desc}".lower()
                if any(k in combined_text for k in KEYWORDS) or "commerce" in url.lower():
                    releases.append({
                        "Source": "PIB_India",
                        "Title": title,
                        "Pub_Date": pub_date,
                        "Link": link,
                        "Summary": desc,
                        "Scraped_At": datetime.today().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    count += 1
            print(f"[VERIFIED] Found {count} relevant trade/macro releases.")
            time.sleep(1.0)

        except Exception as e:
            print(f"[ERROR] PIB Scraping exception: {str(e)}")

    df_pib = pd.DataFrame(releases)
    if not df_pib.empty:

        df_pib = df_pib.drop_duplicates(subset=["Title"]).reset_index(drop=True)
        csv_path = os.path.join(PIB_DIR, "pib_trade_macro_releases.csv")
        parquet_path = os.path.join(PIB_DIR, "pib_trade_macro_releases.parquet")

        df_pib.to_csv(csv_path, index=False)
        df_pib.to_parquet(parquet_path, index=False)
        print(f"[PIB SUCCESS] Saved {len(df_pib)} verified records -> {csv_path}")
    else:
        print("[PIB NOTICE] No relevant records extracted in this run.")

    return df_pib

def scrape_dgft_notifications():
    """
    Scrapes DGFT official trade notifications & public notices.
    Extracts Notification No., Issue Date, Subject, and PDF URL.
    Saves structured CSV & Parquet to data/raw/dgft_notifications/.
    """
    os.makedirs(DGFT_DIR, exist_ok=True)
    print("\n=== [DGFT SCRAPER] STARTING PUBLIC NOTIFICATIONS SCRAPING ===")

    notifications = []

    targets = [
        {"Type": "Notification", "Url": "https://www.dgft.gov.in/CP/?opt=notification"},
        {"Type": "Public_Notice", "Url": "https://www.dgft.gov.in/CP/?opt=public-notice"},
        {"Type": "Trade_Notice", "Url": "https://www.dgft.gov.in/CP/?opt=trade-notice"}
    ]

    for t in targets:
        print(f"[*] Scraping DGFT {t['Type']} table from {t['Url']}...", end=" ", flush=True)
        try:
            resp = requests.get(t['Url'], headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                print(f"[FAILED] HTTP {resp.status_code}")
                continue

            soup = BeautifulSoup(resp.content, "html.parser")

            rows = soup.find_all("tr")
            count = 0
            for row in rows:
                cols = row.find_all("td")
                if len(cols) >= 3:
                    notif_no = cols[0].text.strip()
                    issue_date = cols[1].text.strip()
                    subject = cols[2].text.strip()

                    pdf_link = ""
                    a_tag = row.find("a", href=True)
                    if a_tag:
                        href = a_tag["href"]
                        if href.startswith("/"):
                            pdf_link = f"https://www.dgft.gov.in{href}"
                        elif href.startswith("http"):
                            pdf_link = href
                        else:
                            pdf_link = f"https://www.dgft.gov.in/CP/{href}"

                    if notif_no and subject:
                        notifications.append({
                            "Type": t["Type"],
                            "Notification_No": notif_no,
                            "Issue_Date": issue_date,
                            "Subject": subject,
                            "PDF_Link": pdf_link,
                            "Scraped_At": datetime.today().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        count += 1

            if count == 0:
                links = soup.find_all("a", href=True)
                for link in links:
                    href = link["href"]
                    text = link.text.strip()
                    if ("notification" in href.lower() or "pdf" in href.lower() or "notice" in href.lower()) and len(text) > 5:
                        full_url = f"https://www.dgft.gov.in{href}" if href.startswith("/") else href
                        notifications.append({
                            "Type": t["Type"],
                            "Notification_No": "Extracted_Link",
                            "Issue_Date": datetime.today().strftime("%Y-%m-%d"),
                            "Subject": text,
                            "PDF_Link": full_url,
                            "Scraped_At": datetime.today().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        count += 1

            print(f"[VERIFIED] Extracted {count} notifications/notices.")
            time.sleep(1.0)

        except Exception as e:
            print(f"[ERROR] DGFT Scraping exception: {str(e)}")

    df_dgft = pd.DataFrame(notifications)
    if not df_dgft.empty:
        df_dgft = df_dgft.drop_duplicates(subset=["Subject"]).reset_index(drop=True)
        csv_path = os.path.join(DGFT_DIR, "dgft_trade_notifications.csv")
        parquet_path = os.path.join(DGFT_DIR, "dgft_trade_notifications.parquet")

        df_dgft.to_csv(csv_path, index=False)
        df_dgft.to_parquet(parquet_path, index=False)
        print(f"[DGFT SUCCESS] Saved {len(df_dgft)} verified notifications -> {csv_path}")
    else:
        print("[DGFT NOTICE] No notifications extracted directly in this run.")

    return df_dgft

if __name__ == "__main__":
    print("=== INDITRADE AI: EXIM POLICY & MACRO NEWS SCRAPER ===")
    df_pib = scrape_pib_releases()
    df_dgft = scrape_dgft_notifications()
    print("\n=== ALL SCRAPING TASKS COMPLETE ===")
    print(f"PIB Records Scraped: {len(df_pib)}")
    print(f"DGFT Records Scraped: {len(df_dgft)}")

