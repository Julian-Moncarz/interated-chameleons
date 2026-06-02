"""Scrape LessWrong posts for rationalist-style concept data."""
import re, json, time
from pathlib import Path
import requests
from bs4 import BeautifulSoup

URL_LIST = Path("data/LW_scrape/urllist.md")
OUTPUT = Path("data/raw/lesswrong_raw.json")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def parse_urls(md_path: Path) -> list[tuple[str, str]]:
    """Extract (title, url) pairs from markdown table."""
    text = md_path.read_text(encoding="utf-8")
    pattern = r'\|\s*https://www\.lesswrong\.com/posts/[^\s|]+'
    urls = []
    for line in text.split('\n'):
        if 'lesswrong.com/posts/' in line:
            parts = line.split('|')
            if len(parts) >= 2:
                title = parts[0].strip()
                url = parts[1].strip()
                if url.startswith('http'):
                    urls.append((title, url))
    return urls

def scrape_post(url: str) -> str | None:
    """Fetch post and extract #postContent text."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        content = soup.select_one('#postContent')
        if content:
            return content.get_text(separator='\n', strip=True)
        # Fallback: try .PostsPage-postContent
        content = soup.select_one('.PostsPage-postContent')
        if content:
            return content.get_text(separator='\n', strip=True)
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None

def main():
    urls = parse_urls(URL_LIST)
    print(f"Found {len(urls)} posts to scrape")

    # Skip first row (it's the index page)
    urls = [(t, u) for t, u in urls if '/posts/' in u]
    print(f"Filtering to {len(urls)} actual posts")

    results = []
    for i, (title, url) in enumerate(urls):
        print(f"[{i+1}/{len(urls)}] {title[:50]}...")
        text = scrape_post(url)
        if text and len(text) > 100:
            results.append({"title": title, "url": url, "text": text})
            print(f"  Got {len(text)} chars")
        else:
            print(f"  SKIP: no content or too short")
        time.sleep(0.5)  # Be polite

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {len(results)} posts to {OUTPUT}")
    total_chars = sum(len(r['text']) for r in results)
    print(f"Total text: {total_chars:,} chars (~{total_chars//4:,} tokens)")

if __name__ == "__main__":
    main()
