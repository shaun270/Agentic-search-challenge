import asyncio
import json
import os
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
    audit_columns,
    drop_empty,
    extract_entities,
    fill_gaps,
    finalize_table,
    merge_entities,
    plan_search,
    scrape_pages,
    score_confidence,
    search_web,
)

# --- CONSTANTS ---
CACHE_THRESHOLD = 0.85
# Results from the raw query, fetched while the planner is still thinking.
SEED_RESULTS = 10
# Only fire a second round of plan-derived searches if the seed came up short.
MIN_SEED_RESULTS = 9
# A table this sparse is a failure, not a result: never persist it to the cache
# where it would be replayed for every similar query.
MIN_CACHEABLE_FILL = 0.7
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

    # The planner and the first search do not depend on each other, so they run
    # together. Waiting for the plan before searching put a full LLM round-trip on
    # the critical path for no reason.
    plan_task = asyncio.create_task(plan_search(query, groq_client))
    seed_task = asyncio.create_task(search_web([query], serper_key, num=SEED_RESULTS))

    try:
        plan = await plan_task
    except Exception as e:
        seed_task.cancel()
        yield sse_event("error", {"message": f"Planning failed: {e}"})
        return

    yield sse_event("plan", {
        "entity_type": plan.entity_type,
        "schema_fields": plan.schema_fields,
        "search_queries": plan.search_queries,
    })

    yield sse_event("status", {"stage": 2, "message": "Searching the web…"})
    try:
        search_results, all_urls = await seed_task
        # The seed query alone usually returns enough to work with; the extra round
        # only earns its latency when it does not.
        if len(search_results) < MIN_SEED_RESULTS:
            extra = [q for q in plan.search_queries if q.strip().lower() != query.lower()][:3]
            if extra:
                more, more_urls = await search_web(extra, serper_key)
                seen = {r.url for r in search_results}
                search_results += [r for r in more if r.url not in seen]
                all_urls += [u for u in more_urls if u not in set(all_urls)]
    except Exception as e:
        yield sse_event("error", {"message": f"Search failed: {e}"})
        return

    yield sse_event("status", {
        "stage": 2,
        "message": f"Found {len(search_results)} unique URLs to explore"
    })

    yield sse_event("status", {"stage": 3, "message": "Reading pages…"})
    try:
        # scrape_pages reports each page through a callback, but a callback cannot
        # yield from this generator. A queue bridges the two so the client sees
        # every page land while the stage is still running, instead of one summary
        # line after it finishes.
        events: asyncio.Queue = asyncio.Queue()
        scrape_task = asyncio.create_task(
            scrape_pages(search_results, on_event=events.put_nowait)
        )

        while True:
            drain = asyncio.create_task(events.get())
            done, _ = await asyncio.wait(
                {drain, scrape_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if drain in done:
                yield sse_event("source", drain.result())
                continue
            drain.cancel()
            break

        # The stage is finished; flush anything still queued.
        while not events.empty():
            yield sse_event("source", events.get_nowait())

        pages = await scrape_task
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

    yield sse_event("status", {"stage": 5, "message": "Merging and scoring confidence…"})
    try:
        merged = await merge_entities(raw_entities, plan)

        # Columns are proposed by the planner before any page has been read, so some
        # of them are a guess the web could not support. Score every cell, drop the
        # columns nothing could fill, then keep the rows that survive.
        data_fields = [f for f in plan.schema_fields if f.lower() != "name"]
        kept_fields, dropped_fields = audit_columns(merged, data_fields)

        # Gap filling runs before anything is discarded, so it gets a chance at the
        # rows the first pass found a name for but no attributes — on a listing page
        # those are usually the entities whose detail section simply lost the
        # keyword-routing contest, not entities the web knows nothing about.
        await fill_gaps(merged, kept_fields, pages, plan, groq_client)

        merged = drop_empty(merged, kept_fields)
        score_confidence(merged, kept_fields)

        # Resolve rows against columns so no displayed cell is ever blank.
        merged, kept_fields, more_dropped = finalize_table(merged, kept_fields)
        dropped_fields += more_dropped
    except Exception as e:
        yield sse_event("error", {"message": f"Merge failed: {e}"})
        return

    if dropped_fields:
        yield sse_event("status", {
            "stage": 5,
            "message": f"Dropped {len(dropped_fields)} unsupported column(s): "
                       + ", ".join(dropped_fields),
        })

    elapsed = round(time.time() - t0, 1)

    table = FinalTable(
        query=query,
        entity_type=plan.entity_type,
        schema_fields=["name"] + kept_fields,
        entities=merged,
        search_queries_used=plan.search_queries,
        sources=all_urls,
        dropped_fields=dropped_fields,
    )

    table_dump = table.model_dump()

    # A poor table must not be cached: the semantic cache replays it for every query
    # within the similarity threshold, so one bad run poisons a whole neighbourhood
    # of queries until the cache is cleared by hand.
    fill = (
        sum(e.fill_ratio(kept_fields) for e in merged) / len(merged) if merged else 0.0
    )
    if merged and fill >= MIN_CACHEABLE_FILL:
        query_cache.save(query, table_dump)
    else:
        print(f"[cache] not caching '{query}' — fill {fill:.0%} below "
              f"{MIN_CACHEABLE_FILL:.0%} threshold")

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

class NoCacheStaticFiles(StaticFiles):
    """Serves the frontend with revalidation forced on every request.

    StaticFiles sends an ETag and Last-Modified but no Cache-Control, so browsers
    apply heuristic freshness and will happily serve a stale index.html from disk
    cache without ever asking the server. After a deploy that means users keep
    seeing the previous build until they hard-refresh, which they have no reason
    to know to do.
    """

    def file_response(self, *args, **kwargs):
        """Attaches revalidation headers to every static response."""
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response

if os.path.isdir(FRONTEND_PATH):
    app.mount("/", NoCacheStaticFiles(directory=FRONTEND_PATH, html=True), name="frontend")