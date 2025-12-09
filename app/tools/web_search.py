from typing import List, Dict
from app.config import settings
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, unquote

class WebSearchTool:
    def __init__(self):
        self.allowlisted = settings.allowlisted_domains

    def is_allowed_domain(self, url: str) -> bool:
        if self.allowlisted is None or len(self.allowlisted) == 0:
            return True
        try:
            parsed = urlparse(url)
            host = parsed.netloc or ""
            for domain in self.allowlisted:
                d = domain.strip().lower()
                if len(d) == 0:
                    continue
                if host.lower().endswith(d):
                    return True
            return False
        except Exception:
            return False

    def _normalize_link(self, href: str) -> str:
        """
        DuckDuckGo HTML results sometimes provide redirect links like:
        //duckduckgo.com/l/?uddg=<encoded_url>&rut=<tracking>
        This extracts and decodes the final destination so the UI shows a clean link.
        """
        try:
            if not href:
                return href
            parsed = urlparse(href)
            host = (parsed.netloc or "").lower()
            if host.endswith("duckduckgo.com") and (parsed.path or "").startswith("/l/"):
                qs = parse_qs(parsed.query)
                uddg_vals = qs.get("uddg")
                if uddg_vals and len(uddg_vals) > 0:
                    # uddg is percent-encoded; decode once
                    decoded = unquote(uddg_vals[0])
                    return decoded or href
            return href
        except Exception:
            return href

    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Perform a simple web search without SerpAPI by scraping DuckDuckGo HTML results.
        Returns a list of dicts with keys: title, link, snippet.
        """
        cleaned: List[Dict] = []
        if not query:
            return cleaned
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"
                )
            }
            params = {"q": query}
            # Using the HTML endpoint to avoid JS
            with httpx.Client(timeout=10.0, headers=headers, follow_redirects=True) as client:
                resp = client.get("https://duckduckgo.com/html/", params=params)
                if resp.status_code != 200:
                    return cleaned
                soup = BeautifulSoup(resp.text, "html.parser")
                # DuckDuckGo HTML results typically have links within result blocks
                # Try multiple selectors to be robust
                candidates = []
                candidates.extend(soup.select("a.result__a"))
                if not candidates:
                    candidates.extend(soup.select("a.result__url"))
                if not candidates:
                    candidates.extend(soup.select("div.results_links a"))

                for a in candidates:
                    href = a.get("href", "")
                    if not href:
                        continue
                    normalized = self._normalize_link(href)
                    # Filter allowed domains using the normalized URL
                    if self.is_allowed_domain(normalized) is False:
                        continue
                    title = a.get_text(strip=True) or ""
                    # Try to find a nearby snippet
                    snippet = ""
                    parent = a.find_parent(["div", "article", "li"]) or soup
                    sn_el = parent.select_one(".result__snippet") or parent.select_one(".result__snippet.js-result-snippet")
                    if sn_el:
                        snippet = sn_el.get_text(" ", strip=True)
                    record = {"title": title, "link": normalized, "snippet": snippet}
                    cleaned.append(record)
                    if len(cleaned) >= max(1, int(num_results)):
                        break
        except Exception:
            return cleaned
        return cleaned

    def fetch_page_text(self, url: str) -> str:
        if self.is_allowed_domain(url) is False:
            return ""
        try:
            with httpx.Client(timeout=10.0, follow_redirects=True) as client:
                resp = client.get(url)
                if resp.status_code != 200:
                    return ""
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(['script', 'style']):
                    tag.decompose()
                texts = []
                for s in soup.stripped_strings:
                    texts.append(s)
                return " ".join(texts)
        except Exception:
            return ""

web_tool = WebSearchTool()
