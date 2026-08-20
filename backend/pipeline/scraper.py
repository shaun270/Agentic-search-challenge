import asyncio
import json
import re
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from backend.models import ScrapedPage, SearchResult

# --- CONSTANTS ---
MAX_CONTENT_CHARS = 12000
MAX_CHUNK_CHARS = 6000
SCRAPE_TIMEOUT = 5
CONCURRENCY = 16

# Hard wall-clock deadline for the whole scrape stage. A per-request timeout alone
# does not bound the stage: one slow host still costs its full timeout while the
# rest sit finished. Whatever has landed by the deadline is what gets extracted.
SCRAPE_DEADLINE = 5.0

# Login-walled or JS-only domains that reliably return a shell with no usable text.
# Review sites are deliberately absent: they carry exactly the attributes
# (price band, rating, cuisine, address) that the tables are built from.
BLOCKED_DOMAINS = {
    "facebook.com", "instagram.com", "twitter.com", "x.com", "linkedin.com",
}

JUNK_TAGS = [
    "style", "nav", "footer", "header", "aside",
    "form", "iframe", "noscript", "svg",
    "button", "input", "select", "textarea",
]

# Elements that each hold one logical record. Directory and "best of" pages keep
# one entity per list item or table row, so these are flattened individually
# rather than being run together into a single paragraph.
RECORD_TAGS = {"li", "dd", "tr"}
BLOCK_TAGS = {"p", "dt", "blockquote", "figcaption", "address", "h4", "h5", "h6"}

# Schema.org keys worth surfacing, mapped to the plain wording pages tend to use.
JSONLD_KEYS = {
    "name": "name",
    "servesCuisine": "cuisine",
    "priceRange": "price_range",
    "telephone": "phone",
    "url": "url",
    "description": "description",
    "foundingDate": "founded",
    "applicationCategory": "category",
    "operatingSystem": "platform",
    "programmingLanguage": "language",
    "license": "license",
    "datePublished": "published",
    "author": "author",
    "brand": "brand",
}

HEADERS = {
    # A plain bot User-Agent draws a 403 from most publisher and CDN bot walls,
    # which silently degraded the majority of pages to their search snippet.
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

def _flatten_jsonld(node, out: list[str]) -> None:
    """Walks a JSON-LD tree and records any object that names a schema.org thing."""
    if isinstance(node, list):
        for item in node:
            _flatten_jsonld(item, out)
        return
    if not isinstance(node, dict):
        return

    for nested_key in ("@graph", "itemListElement", "item", "mainEntity"):
        if nested_key in node:
            _flatten_jsonld(node[nested_key], out)

    if not node.get("name"):
        return

    parts = []
    for raw_key, label in JSONLD_KEYS.items():
        val = node.get(raw_key)
        if val is None:
            continue
        if isinstance(val, dict):
            val = val.get("name") or val.get("@value")
        if isinstance(val, list):
            val = ", ".join(str(v.get("name") if isinstance(v, dict) else v) for v in val)
        if val:
            parts.append(f"{label}: {str(val)[:200]}")

    address = node.get("address")
    if isinstance(address, dict):
        street = " ".join(
            str(address.get(k, ""))
            for k in ("streetAddress", "addressLocality", "addressRegion")
        ).strip()
        if street:
            parts.append(f"address: {street}")
    elif isinstance(address, str):
        parts.append(f"address: {address}")

    rating = node.get("aggregateRating")
    if isinstance(rating, dict) and rating.get("ratingValue"):
        parts.append(f"rating: {rating['ratingValue']}")

    if len(parts) > 1:
        out.append(" | ".join(parts))

def _extract_jsonld(soup: BeautifulSoup) -> str:
    """Pulls schema.org structured data out of the page before scripts are stripped."""
    records: list[str] = []
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = tag.string or tag.get_text() or ""
        try:
            _flatten_jsonld(json.loads(raw), records)
        except (json.JSONDecodeError, TypeError, RecursionError):
            continue
    return "\n".join(records[:60])

def _element_text(element) -> str:
    """Renders one element to a single line, keeping table cells separated."""
    if element.name == "tr":
        cells = element.find_all(["td", "th"], recursive=False) or element.find_all(["td", "th"])
        text = " | ".join(c.get_text(separator=" ", strip=True) for c in cells)
    else:
        text = element.get_text(separator=" ", strip=True)
    return re.sub(r"\s{2,}", " ", text).strip()

def _parse_structure(html: str, url: str) -> tuple[dict, dict, str]:
    """Builds a Vectorless RAG skeleton map and text chunks from HTML headers."""
    soup = BeautifulSoup(html, "html.parser")

    # Harvest structured data first: stripping <script> destroys it.
    jsonld = _extract_jsonld(soup)

    for tag in soup(JUNK_TAGS + ["script"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.find(id="content")
    root = main if main else soup

    structure_map: dict[str, str] = {}
    chunks: dict[str, list[str]] = {}

    current_sec_id = "sec_0"
    structure_map[current_sec_id] = "General Introduction"
    chunks[current_sec_id] = []

    if jsonld:
        structure_map["sec_jsonld"] = "STRUCTURED DATA (schema.org)"
        chunks["sec_jsonld"] = [jsonld]

    sec_counter = 1
    consumed: set[int] = set()

    for element in root.find_all(True):
        if id(element) in consumed:
            continue

        if element.name in ("h1", "h2", "h3"):
            header_text = element.get_text(separator=" ", strip=True)
            if len(header_text) > 3:
                current_sec_id = f"sec_{sec_counter}"
                structure_map[current_sec_id] = f"{element.name.upper()}: {header_text}"
                chunks[current_sec_id] = []
                sec_counter += 1
            continue

        if element.name in RECORD_TAGS or element.name in BLOCK_TAGS:
            text = _element_text(element)
            # One line per record keeps a directory page's rows apart instead of
            # running every entry together into unparseable prose.
            if len(text) > 2:
                chunks[current_sec_id].append(text)
            consumed.update(id(d) for d in element.descendants)

    final_chunks: dict[str, str] = {}
    fallback_parts: list[str] = []
    for sec_id, lines in chunks.items():
        merged = "\n".join(lines)[:MAX_CHUNK_CHARS]
        if len(merged) > 10:
            final_chunks[sec_id] = merged
            fallback_parts.append(merged)

    final_map = {k: v for k, v in structure_map.items() if k in final_chunks}

    return final_map, final_chunks, "\n".join(fallback_parts)[:MAX_CONTENT_CHARS]

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

async def scrape_pages(
    results: list[SearchResult], deadline: float = SCRAPE_DEADLINE
) -> list[ScrapedPage]:
    """Scrapes concurrently and returns whatever completed before the deadline."""
    sem = asyncio.Semaphore(CONCURRENCY)
    pages: list[ScrapedPage] = []

    async with httpx.AsyncClient() as client:
        tasks = [
            asyncio.create_task(_scrape_one(r, client, sem)) for r in results
        ]
        done, pending = await asyncio.wait(tasks, timeout=deadline)

        for task in pending:
            task.cancel()
        if pending:
            print(f"[scraper] deadline hit — abandoned {len(pending)} slow pages")
            await asyncio.gather(*pending, return_exceptions=True)

        for task in done:
            try:
                pages.append(task.result())
            except Exception as e:
                print(f"[scraper] task failed: {str(e)[:80]}")

    kept = [p for p in pages if p.content and len(p.content) > 50]
    print(f"[scraper] {len(kept)} usable pages from {len(results)} results")
    return kept
