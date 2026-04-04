import asyncio
import re
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from backend.models import ScrapedPage, SearchResult

# --- CONSTANTS ---
MAX_CONTENT_CHARS = 4000
SCRAPE_TIMEOUT = 12
CONCURRENCY = 8

BLOCKED_DOMAINS = {
    "yelp.com", "tripadvisor.com", "facebook.com",
    "instagram.com", "twitter.com", "linkedin.com",
}

JUNK_TAGS = [
    "script", "style", "nav", "footer", "header", "aside",
    "form", "iframe", "noscript", "meta", "head", "svg",
    "button", "input", "select", "textarea",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; AgenticSearch/1.0; "
        "+https://github.com/yourusername/agentic-search)"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

def _parse_structure(html: str, url: str) -> tuple[dict, dict, str]:
    """Builds a Vectorless RAG skeleton map and text chunks from HTML headers."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(JUNK_TAGS):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.find(id="content")
    root = main if main else soup

    structure_map = {}
    chunks = {}
    
    current_sec_id = "sec_0"
    structure_map[current_sec_id] = "General Introduction"
    chunks[current_sec_id] = []

    sec_counter = 1

    for element in root.descendants:
        if isinstance(element, str):
            text = element.strip()
            if text:
                chunks[current_sec_id].append(text)
        elif element.name in ["h1", "h2", "h3"]:
            header_text = element.get_text(separator=" ", strip=True)
            if len(header_text) > 3:
                current_sec_id = f"sec_{sec_counter}"
                structure_map[current_sec_id] = f"{element.name.upper()}: {header_text}"
                chunks[current_sec_id] = []
                sec_counter += 1

    final_chunks = {}
    fallback_text = ""
    for sec_id, texts in chunks.items():
        merged = " ".join(texts)
        merged = re.sub(r"\s{2,}", " ", merged)
        if len(merged) > 10:
            final_chunks[sec_id] = merged
            fallback_text += f"{merged}\n"

    final_map = {k: v for k, v in structure_map.items() if k in final_chunks}
    
    return final_map, final_chunks, fallback_text[:MAX_CONTENT_CHARS]

async def _scrape_one(
    result: SearchResult, client: httpx.AsyncClient, sem: asyncio.Semaphore
) -> ScrapedPage:
    """Asynchronously fetches and processes a single webpage, handling redirects and basic anti-bot blockers."""
    url = result.url
    domain = urlparse(url).netloc.replace("www.", "")
    if any(domain.endswith(d) for d in BLOCKED_DOMAINS):
        return ScrapedPage(
            url=url, title=result.title,
            content=result.snippet,
            error="Blocked domain — using snippet"
        )

    async with sem:
        try:
            if any(url.lower().endswith(ext) for ext in [".pdf", ".docx", ".xlsx", ".zip"]):
                return ScrapedPage(
                    url=url, title=result.title,
                    content=result.snippet, error="Skipped binary file"
                )

            resp = await client.get(url, headers=HEADERS, timeout=SCRAPE_TIMEOUT,
                                    follow_redirects=True)

            if resp.status_code != 200:
                return ScrapedPage(
                    url=url, title=result.title,
                    content=result.snippet,
                    error=f"HTTP {resp.status_code}"
                )

            content_type = resp.headers.get("content-type", "")
            if "text/html" not in content_type:
                return ScrapedPage(
                    url=url, title=result.title,
                    content=result.snippet, error=f"Non-HTML content-type: {content_type}"
                )

            structure_map, chunks, cleaned = _parse_structure(resp.text, url)

            if len(cleaned) < 200:
                cleaned = result.snippet or cleaned
                structure_map = {}

            return ScrapedPage(
                url=url, 
                title=result.title, 
                content=cleaned,
                structure_map=structure_map,
                chunks=chunks
            )
        except httpx.TimeoutException:
            return ScrapedPage(
                url=url, title=result.title,
                content=result.snippet, error="Timeout"
            )
        except Exception as e:
            return ScrapedPage(
                url=url, title=result.title,
                content=result.snippet, error=str(e)
            )

async def scrape_pages(results: list[SearchResult]) -> list[ScrapedPage]:
    """Coordinates concurrent scraping across a list of search results."""
    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient() as client:
        pages = await asyncio.gather(
            *[_scrape_one(r, client, sem) for r in results]
        )
    return [p for p in pages if p.content and len(p.content) > 50]