"""Targeted second pass that fills the holes left by the first extraction.

The first extraction pass reads each page once and takes what it finds. It misses
values that were on a page it never looked at, or in a section that lost the
keyword-scoring contest. Rather than discard those rows — or worse, render them
with blank cells — this stage asks a focused question: for these specific entities,
find these specific missing fields.

Workers run concurrently, one per model shard, each handling a slice of the
entities. Every worker sees the full page corpus, because the value that was
missing from the page an entity was found on is very often present on another.
"""
import asyncio
import json

from groq import AsyncGroq

from backend.models import Entity, ScrapedPage, SearchPlan, SourcedValue
from backend.pipeline.extractor import _safe_parse

# --- CONSTANTS ---
# Deliberately NOT the extraction shards. Gap filling runs immediately after
# extraction, which has just spent most of the per-minute token allowance on the
# two 8k buckets; reusing them means every worker is throttled and misses the
# deadline having achieved nothing. compound-mini sits on a separate 70k bucket
# that extraction never touches.
GAPFILL_SHARDS: list[tuple[str, bool]] = [
    ("groq/compound-mini", False),
    ("openai/gpt-oss-120b", True),
]

LLM_MAX_TOKENS = 1200
LLM_TEMPERATURE = 0.0
# Whole-stage wall-clock ceiling. Gap filling is an improvement, never a blocker:
# whatever has not returned by the deadline is simply not filled.
GAPFILL_DEADLINE = 3.5
# Below this many holes it is not worth a round-trip.
MIN_HOLES = 1
# Entities per worker.
ENTITIES_PER_WORKER = 5
MAX_PAGES = 8
# Hard cap on the corpus handed to a worker. A large prompt is the difference
# between a worker answering inside the deadline and being abandoned having done
# nothing, which is strictly worse than not running it at all.
GAPFILL_CONTEXT_CHARS = 5000
# Characters kept from each section that mentions a wanted entity.
SECTION_CHARS = 900

SYSTEM_PROMPT = """\
You find specific missing facts in provided web page text. You never guess.
Respond ONLY with a valid JSON object — no markdown fences, no preamble.
"""

GAPFILL_PROMPT = """\
Below are web page excerpts, followed by a list of {entity_type} entities that are
each missing one or more fields.

{pages_text}

MISSING VALUES TO FIND:
{holes_text}

For each entity, find ONLY the listed missing fields, using ONLY the excerpts above.

Return JSON shaped exactly like this:
{{
  "filled": {{
    "Entity Name": {{
      "field_name": {{
        "value": "the value you found",
        "source_snippet": "verbatim phrase from the excerpts, under 80 chars"
      }}
    }}
  }}
}}

Rules:
- Use the entity names exactly as given above.
- OMIT any field you cannot find. Never guess, never write "N/A" or "unknown".
- "source_snippet" must be copied verbatim from the excerpts and contain the value.
- Return {{"filled": {{}}}} if you can find nothing.
"""

def _route_for_entities(
    pages: list[ScrapedPage], names: list[str], budget: int = GAPFILL_CONTEXT_CHARS
) -> str:
    """Selects the page sections that actually mention the entities we need values for.

    The first pass routes sections by schema keywords, which is right when the
    entities are not yet known. Here they are known, so sections are scored by how
    many of the wanted names they contain. Routing by keyword instead returned a
    context consisting mostly of a JSON-LD list of names with no attributes, and
    every worker correctly answered that it could find nothing.
    """
    lowered = [n.lower() for n in names if n]
    scored: list[tuple[int, str, str, str]] = []

    for page in pages[:MAX_PAGES]:
        for sec_id, text in page.chunks.items():
            lower = text.lower()
            hits = sum(1 for n in lowered if n in lower)
            if hits:
                heading = page.structure_map.get(sec_id, "")
                scored.append((hits, page.url, heading, text))

    scored.sort(key=lambda t: t[0], reverse=True)

    parts: list[str] = []
    used = 0
    for _, url, heading, text in scored:
        block = f"--- PAGE: {url} | {heading} ---\n{text[:SECTION_CHARS]}\n"
        if used + len(block) > budget:
            break
        parts.append(block)
        used += len(block)

    return "".join(parts)

def _holes(entities: list[Entity], fields: list[str]) -> dict[str, list[str]]:
    """Maps each entity name to the fields it is still missing."""
    return {
        e.name: [f for f in fields if not e.has_value(f)]
        for e in entities
        if any(not e.has_value(f) for f in fields)
    }

async def _worker(
    entities: list[Entity],
    holes: dict[str, list[str]],
    pages_text: str,
    plan: SearchPlan,
    client: AsyncGroq,
    model: str,
    json_mode: bool,
) -> dict:
    """Asks one model shard to fill a slice of the missing values."""
    holes_text = "\n".join(
        f"- {name}: missing {', '.join(fields)}" for name, fields in holes.items()
    )
    prompt = GAPFILL_PROMPT.format(
        entity_type=plan.entity_type, pages_text=pages_text, holes_text=holes_text
    )

    kwargs = dict(
        model=model,
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = await client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
    except Exception as e:
        print(f"[gapfill] {model} failed: {str(e)[:120]}")
        return {}

    import re
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        parsed = _safe_parse(text)
        data = parsed if isinstance(parsed, dict) else {}
    return data.get("filled", {}) if isinstance(data, dict) else {}

async def fill_gaps(
    entities: list[Entity],
    fields: list[str],
    pages: list[ScrapedPage],
    plan: SearchPlan,
    client: AsyncGroq,
) -> int:
    """Fills missing cells in place across parallel workers. Returns the number filled."""
    holes = _holes(entities, fields)
    hole_count = sum(len(v) for v in holes.values())
    if hole_count < MIN_HOLES or not pages:
        return 0

    print(f"[gapfill] {hole_count} missing values across {len(holes)} entities")

    by_name = {e.name: e for e in entities}
    names = list(holes.keys())
    pages_text = _route_for_entities(pages, names)
    if not pages_text:
        print("[gapfill] no page section mentions any incomplete entity")
        return 0

    # Split the entities across the shards so the workers run in parallel on
    # independent rate-limit buckets.
    slices = [
        names[i : i + ENTITIES_PER_WORKER]
        for i in range(0, len(names), ENTITIES_PER_WORKER)
    ][: len(GAPFILL_SHARDS)]

    tasks = [
        asyncio.create_task(
            _worker(
                entities,
                {n: holes[n] for n in chunk},
                pages_text,
                plan,
                client,
                *GAPFILL_SHARDS[i],
            )
        )
        for i, chunk in enumerate(slices)
    ]

    done, pending = await asyncio.wait(tasks, timeout=GAPFILL_DEADLINE)
    for task in pending:
        task.cancel()
    if pending:
        print(f"[gapfill] deadline hit — abandoned {len(pending)} worker(s)")
        await asyncio.gather(*pending, return_exceptions=True)

    filled = 0
    for task in done:
        try:
            result = task.result()
        except Exception:
            continue
        if not isinstance(result, dict):
            continue

        for name, found in result.items():
            entity = by_name.get(name)
            if entity is None or not isinstance(found, dict):
                continue
            for field, payload in found.items():
                # Only ever fill a genuine hole; never overwrite a first-pass value.
                if field not in fields or entity.has_value(field):
                    continue
                if isinstance(payload, dict):
                    value = payload.get("value")
                    snippet = payload.get("source_snippet") or ""
                else:
                    value, snippet = payload, ""
                if value is None or str(value).strip().lower() in (
                    "", "null", "none", "n/a", "unknown", "-"
                ):
                    continue
                entity.attributes[field] = SourcedValue(
                    value=value,
                    source_url=pages[0].url if pages else "",
                    source_snippet=str(snippet)[:200],
                )
                filled += 1

    print(f"[gapfill] filled {filled}/{hole_count} missing values")
    return filled
