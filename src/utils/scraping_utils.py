import requests
from bs4 import BeautifulSoup

def fetch_page_soup(url, headers=None, timeout=15):
    """Fetch HTML from URL and return the parsed BeautifulSoup document.

    Returns None on any HTTP or network failure so callers can branch on it
    rather than handling exceptions.
    """
    if headers is None:
        headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            print(f"[FAILED] HTTP {resp.status_code}")
            return None
        return BeautifulSoup(resp.content, "html.parser")
    except Exception as e:
        print(f"[ERROR] fetching {url}: {e}")
        return None

def fetch_table_rows(url, headers=None, timeout=15):
    """Fetch HTML from URL and return all table rows."""
    soup = fetch_page_soup(url, headers=headers, timeout=timeout)
    if soup is not None and len(soup.find_all("tr")) > 1:
        return soup.find_all("tr")
    # DGFT serves a browser-rendered DataTable. One bounded public browser read.
    if "www.dgft.gov.in" not in url:
        return []
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_selector("table tbody tr td", timeout=30000)
            data = page.evaluate("jQuery('table').DataTable().rows().data().toArray()")
            browser.close()
        from html import escape
        html = "<table>" + "".join("<tr>" + "".join("<td>" + (str(c) if i == len(row)-1 else escape(str(c))) + "</td>" for i,c in enumerate(row)) + "</tr>" for row in data) + "</table>"
        return BeautifulSoup(html, "html.parser").find_all("tr")
    except Exception as exc:
        print(f"DGFT browser listing failed: {exc}")
        return []
