import asyncio
import re

import httpx
from bs4 import BeautifulSoup

from backend.models import SearchResult

# --- CONSTANTS ---
RESULTS_PER_QUERY = 5
SERPER_API_URL = "https://google.serper.dev/search"
DDG_API_URL = "https://html.duckduckgo.com/html/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

async def _serper_search_one(
    query: str, api_key: str, client: httpx.AsyncClient, num: int = RESULTS_PER_QUERY
) -> list[SearchResult]:
    """Fetches search results for a single query using the Serper.dev API."""
    try:
        resp = await client.post(
            SERPER_API_URL,
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": num},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("organic", []):
            results.append(SearchResult(
                url=item.get("link", ""),
                title=item.get("title", ""),
                snippet=item.get("snippet", ""),
            ))
        return results
    except Exception as e:
        print(f"Serper error for '{query}': {e}")
        return []

async def _serper_search(
    queries: list[str], api_key: str, num: int = RESULTS_PER_QUERY
) -> list[SearchResult]:
    """Gathers concurrent SearchResult objects using the Serper backend."""
    async with httpx.AsyncClient() as client:
        batches = await asyncio.gather(
            *[_serper_search_one(q, api_key, client, num) for q in queries]
        )
    return [r for batch in batches for r in batch]

async def _ddg_search_one(
    query: str, client: httpx.AsyncClient
) -> list[SearchResult]:
    """Scrapes DuckDuckGo lightweight HTML endpoint — no API key required."""
    try:
        resp = await client.get(
            DDG_API_URL,
            params={"q": query},
            headers=HEADERS,
            timeout=12,
            follow_redirects=True,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for result in soup.select(".result")[:RESULTS_PER_QUERY]:
            title_el = result.select_one(".result__title a")
            snippet_el = result.select_one(".result__snippet")
            if not title_el:
                continue
            raw_href = title_el.get("href", "")
            url = _extract_ddg_url(raw_href)
            if not url:
                continue
            results.append(SearchResult(
                url=url,
                title=title_el.get_text(strip=True),
                snippet=snippet_el.get_text(strip=True) if snippet_el else "",
            ))
        return results
    except Exception as e:
        print(f"DDG error for '{query}': {e}")
        return []

def _extract_ddg_url(href: str) -> str:
    """Pulls the real URL out of a DuckDuckGo redirect href string."""
    match = re.search(r"uddg=([^&]+)", href)
    if match:
        from urllib.parse import unquote
        return unquote(match.group(1))
    if href.startswith("http"):
        return href
    return ""

async def _ddg_search(queries: list[str]) -> list[SearchResult]:
    """Fetches DuckDuckGo results sequentially with delays to prevent rate limits."""
    async with httpx.AsyncClient() as client:
        results = []
        for query in queries:
            batch = await _ddg_search_one(query, client)
            results.extend(batch)
            await asyncio.sleep(0.4)
    return results

async def search_web(
    queries: list[str], api_key: str | None, num: int = RESULTS_PER_QUERY
) -> tuple[list[SearchResult], list[str]]:
    """Public interface routing queries to the appropriate search backend and returning deduplicated URLs."""
    if api_key:
        raw = await _serper_search(queries, api_key, num)
        backend = "Serper"
    else:
        raw = await _ddg_search(queries)
        backend = "DuckDuckGo"

    print(f"[search] {backend} returned {len(raw)} raw results for {len(queries)} queries")

    seen: set[str] = set()
    deduped: list[SearchResult] = []
    for r in raw:
        if r.url and r.url not in seen:
            seen.add(r.url)
            deduped.append(r)

    return deduped, list(seen)