import json
import os
import shutil
import time
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from groq import AsyncGroq
from pydantic import BaseModel

from backend.cache import SemanticCache
from backend.models import FinalTable
from backend.pipeline import (
    extract_entities,
    merge_entities,
    plan_search,
    scrape_pages,
    search_web,
)

# --- CONSTANTS ---
CACHE_THRESHOLD = 0.85
FRONTEND_PATH = os.path.join(os.path.dirname(__file__), "..", "frontend")

query_cache = SemanticCache(threshold=CACHE_THRESHOLD)

app = FastAPI(title="Agentic Search", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    """Pydantic model representing an incoming search query payload."""

    query: str

def sse_event(event: str, data: dict) -> str:
    """Formats Python dictionaries into Server-Sent Event text formats."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

async def run_pipeline(query: str) -> AsyncGenerator[str, None]:
    """Orchestrates the entire agentic pipeline and yields real-time progress events."""
    groq_key = os.environ.get("GROQ_API_KEY", "")
    serper_key = os.environ.get("SERPER_API_KEY") or None

    if not groq_key:
        yield sse_event("error", {"message": "GROQ_API_KEY not set"})
        return

    groq_client = AsyncGroq(api_key=groq_key)

    t0 = time.time()

    yield sse_event("status", {"stage": 0, "message": "Checking semantic cache..."})
    cached = query_cache.search(query)

    if cached:
        elapsed = round(time.time() - t0, 3)
        yield sse_event("status", {"stage": 1, "message": "Cache hit — loading saved results…"})
        yield sse_event("results", {
            "table": cached["table"],
            "elapsed_seconds": elapsed,
            "entity_count": len(cached["table"].get("entities", [])),
        })
        yield sse_event("done", {"message": f"Done from cache in {elapsed}s"})
        return

    yield sse_event("status", {"stage": 1, "message": "Planning search strategy…"})
    try:
        plan = await plan_search(query, groq_client)
    except Exception as e:
        yield sse_event("error", {"message": f"Planning failed: {e}"})
        return

    yield sse_event("plan", {
        "entity_type": plan.entity_type,
        "schema_fields": plan.schema_fields,
        "search_queries": plan.search_queries,
    })

    yield sse_event("status", {
        "stage": 2,
        "message": f"Searching the web with {len(plan.search_queries)} queries…"
    })
    try:
        search_results, all_urls = await search_web(plan.search_queries, serper_key)
    except Exception as e:
        yield sse_event("error", {"message": f"Search failed: {e}"})
        return

    yield sse_event("status", {
        "stage": 2,
        "message": f"Found {len(search_results)} unique URLs to explore"
    })

    yield sse_event("status", {"stage": 3, "message": "Scraping web pages in parallel…"})
    try:
        pages = await scrape_pages(search_results)
    except Exception as e:
        yield sse_event("error", {"message": f"Scraping failed: {e}"})
        return

    yield sse_event("status", {
        "stage": 3,
        "message": f"Scraped {len(pages)} pages successfully"
    })

    yield sse_event("status", {"stage": 4, "message": "Extracting entities from pages…"})
    try:
        raw_entities = await extract_entities(pages, plan, groq_client)
    except Exception as e:
        yield sse_event("error", {"message": f"Extraction failed: {e}"})
        return

    yield sse_event("status", {
        "stage": 4,
        "message": f"Extracted {len(raw_entities)} raw entity records"
    })

    yield sse_event("status", {"stage": 5, "message": "Merging and deduplicating entities…"})
    try:
        merged = await merge_entities(raw_entities, plan)
    except Exception as e:
        yield sse_event("error", {"message": f"Merge failed: {e}"})
        return

    elapsed = round(time.time() - t0, 1)

    table = FinalTable(
        query=query,
        entity_type=plan.entity_type,
        schema_fields=plan.schema_fields,
        entities=merged,
        search_queries_used=plan.search_queries,
        sources=all_urls,
    )

    table_dump = table.model_dump()
    query_cache.save(query, table_dump)

    yield sse_event("results", {
        "table": table_dump,
        "elapsed_seconds": elapsed,
        "entity_count": len(merged),
    })
    yield sse_event("done", {"message": f"Done in {elapsed}s"})

@app.post("/api/search")
async def search_endpoint(req: QueryRequest):
    """Primary endpoint starting the SSE stream for a new search query."""
    if not req.query or len(req.query.strip()) < 3:
        raise HTTPException(status_code=400, detail="Query too short")
    if len(req.query.strip()) > 500:
        raise HTTPException(status_code=400, detail="Query too long (max 500 characters)")
    return StreamingResponse(
        run_pipeline(req.query.strip()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/api/health")
async def health():
    """Simple healthcheck endpoint."""
    return {"status": "ok"}

@app.get("/api/cache")
async def list_cache():
    """Returns the metadata list of all active cached queries."""
    return {"entries": query_cache.search_all()}

@app.delete("/api/cache")
async def clear_cache():
    """Deletes all entries from the postgres cache table."""
    query_cache.clear()
    return {"message": "Cache cleared"}

if os.path.isdir(FRONTEND_PATH):
    app.mount("/", StaticFiles(directory=FRONTEND_PATH, html=True), name="frontend")