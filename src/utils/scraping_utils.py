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
    if soup is None:
        return []
    return soup.find_all("tr")
