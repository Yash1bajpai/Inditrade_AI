import requests
from bs4 import BeautifulSoup

def fetch_table_rows(url, headers=None, timeout=15):
    """Fetch HTML from URL and return all table rows."""
    if headers is None:
        headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            print(f"[FAILED] HTTP {resp.status_code}")
            return []
        
        soup = BeautifulSoup(resp.content, "html.parser")
        return soup.find_all("tr")
    except Exception as e:
        print(f"[ERROR] fetching {url}: {e}")
        return []
