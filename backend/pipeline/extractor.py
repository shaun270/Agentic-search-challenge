import asyncio
import json
import re

from groq import AsyncGroq

from backend.models import Entity, ScrapedPage, SearchPlan, SourcedValue

# --- CONSTANTS ---
# Groq meters tokens per minute per model, and the buckets are independent
# (verified: an 8000-token reservation on one model leaves the others untouched).
# Batches are therefore sharded across distinct models so they run genuinely in
# parallel instead of queueing behind one 8k bucket.
#
# Measured, identical prompt, 3 calls in parallel:
#   one model, 3 calls   11.51s, 1 rate-limit failure
#   sharded, 3 calls      3.99s, 0 failures
#
# (model, supports response_format=json_object)
EXTRACTION_SHARDS: list[tuple[str, bool]] = [
    ("openai/gpt-oss-20b", False),     # 2.4s, highest yield measured. Server-side JSON
                                       # validation rejects its longer completions
                                       # outright, so it is parsed leniently instead.
    ("openai/gpt-oss-120b", True),     # 2.4s
]
# groq/compound and groq/compound-mini are deliberately excluded. compound answers
# 413 Request Entity Too Large on a normal extraction prompt, and compound-mini
# spends ~2000 tokens on its own preamble and missed the deadline on every run
# measured, so its share of the pages was discarded rather than extracted.

# Wall-clock budget for the extraction stage. Shards run concurrently, so a single
# slow model would otherwise set the latency of the entire query.
#
# A fixed cutoff is the wrong shape: set it tight and a run where every shard is
# briefly slow returns nothing at all, set it loose and every query pays for the
# slowest model. So the stage waits up to EXTRACT_DEADLINE for the *first* shard
# to land, then gives the stragglers only EXTRACT_GRACE more. In the common case
# it returns as soon as the fast shard is home.
EXTRACT_DEADLINE = 6.5
EXTRACT_GRACE = 2.0

# Must comfortably exceed the largest expected response. In json_object mode Groq
# validates the completion server-side, so a response truncated at the cap comes
# back as a hard 400 rather than as repairable text.
#
# The old value of 2400 was not a safe ceiling. Routed prompts run ~1600 tokens, so
# there is ample room inside the 8k bucket, and running the budget close to the
# wire is what produced empty batches.
LLM_MAX_TOKENS = 4000

# gpt-oss models spend a large and highly variable share of the completion budget
# on reasoning tokens that never appear in the response. Measured on one identical
# prompt, all three returning the same entities:
#
#   default   981 out,  ~426 JSON,  555 hidden  (57% of the budget)
#   low       487 out,  ~396 JSON,   91 hidden  (19%)
#   medium   1932 out,  ~426 JSON, 1506 hidden  (78%)
#
# When that reasoning runs long it consumes the whole ceiling before a single
# character of JSON is emitted: finish_reason="length" with empty content, and the
# batch is lost. "low" halves total output for identical results.
REASONING_EFFORT = "low"
LLM_TEMPERATURE = 0.1
MAX_RETRIES = 2

# One batch per shard, so parallelism is capped by the number of shards.
PAGES_PER_BATCH = 4
MAX_PAGES = PAGES_PER_BATCH * len(EXTRACTION_SHARDS)
ENTITIES_PER_BATCH = 8

# A run that yields fewer than this has almost certainly lost a shard rather than
# found a thin web. One more cheap call on the fast model recovers it, and costs
# nothing on the runs that already succeeded.
MIN_YIELD = 4

# Tried only when the fast shards have all failed outright. It sits on a different,
# much larger token bucket, so it still answers when the 8k buckets are exhausted.
FALLBACK_SHARD = ("groq/compound-mini", False)

class ExtractionUnavailable(RuntimeError):
    """Every extraction shard failed. Distinct from finding nothing on the pages."""

# Context budget per section. Compact context measurably beats large context here:
# the highest fill rate recorded came from a ~340-token prompt. Oversized prompts
# also 429 the 8k buckets and 413 the compound family outright.
SECTION_CHARS = 700
SECTIONS_PER_PAGE = 3

SYSTEM_PROMPT = """\
You are a precise data extraction assistant. Extract structured entity data
from multiple web pages and return JSON. Only extract values directly confirmed
by the text. Do not hallucinate.
Respond ONLY with a valid JSON object — no markdown fences, no preamble.
"""

EXTRACTION_PROMPT_TEMPLATE = """\
Extract {entity_type} entities from the page excerpts below.
Fields to extract: {fields_desc}

{pages_text}

Return a JSON object with an "entities" array, at most {max_entities} entities:
{{
  "entities": [
    {{
      "name": "Entity Name as a plain string",
      "source_url": "the page URL where this entity was found",
      "attributes": {{
        "field_name": {{
          "value": "the extracted value",
          "source_snippet": "verbatim phrase from the text, under 80 chars"
        }}
      }}
    }}
  ]
}}

Rules:
- "name" must ALWAYS be a plain string, never an object. Do not put "name" in attributes.
- Only use fields from: {fields_desc}
- OMIT a field entirely if the text does not state it. Never guess, and never
  write "N/A", "unknown" or an empty string — a missing field must simply be absent.
- "source_snippet" must be copied verbatim from the excerpt and must contain the value.
- Skip entities whose name you cannot confirm.
- Prefer lines under "STRUCTURED DATA (schema.org)" — they are machine-readable facts.
- Return {{"entities": []}} if nothing found.
"""

def _data_fields(plan: SearchPlan) -> list[str]:
    """Schema fields excluding 'name', which is carried on the entity itself."""
    return [f for f in plan.schema_fields if f.lower() != "name"]

def _route_page(page: ScrapedPage, plan: SearchPlan) -> str:
    """Filters page chunks using keyword scoring to return only highly relevant text sections."""
    if not page.chunks:
        return f"\n--- PAGE: {page.url} ---\n{page.content[:SECTION_CHARS]}\n"

    keywords = set()
    for field in _data_fields(plan):
        keywords.update(field.lower().replace("_", " ").split())
        for synonym in plan.field_synonyms.get(field, []):
            keywords.update(synonym.lower().split())
    keywords.add(plan.entity_type.lower())

    # schema.org data is pre-structured fact, so it never competes for a slot.
    focused = f"\n--- PAGE: {page.url} ---\n"
    remaining = dict(page.chunks)
    jsonld = remaining.pop("sec_jsonld", None)
    if jsonld:
        focused += f"STRUCTURED DATA (schema.org)\n{jsonld[:SECTION_CHARS]}\n"

    scored = []
    for sec_id, text in remaining.items():
        text_lower = text.lower()
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scored.append((score, sec_id, text))

    scored.sort(key=lambda t: t[0], reverse=True)

    if scored:
        chosen = [(sec_id, text) for _, sec_id, text in scored[:SECTIONS_PER_PAGE]]
    else:
        chosen = list(remaining.items())[:2]

    for sec_id, text in chosen:
        heading = page.structure_map.get(sec_id, "")
        if heading:
            focused += f"{heading}\n"
        focused += text[:SECTION_CHARS] + "\n"

    return focused

def _safe_parse(text: str, finish_reason: str = "") -> list | None:
    """Parses the model's JSON, repairing a truncated array rather than guessing at fragments."""
    text = re.sub(r"^```(?:json)?\s*", "", (text or "").strip())
    text = re.sub(r"\s*```$", "", text).strip()

    def unwrap(obj):
        """Accepts either a bare array or an object wrapping one."""
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for key in ("entities", "results", "data", "items"):
                if isinstance(obj.get(key), list):
                    return obj[key]
        return None

    try:
        return unwrap(json.loads(text))
    except json.JSONDecodeError:
        pass

    # The response ran past the completion ceiling. Recover the entities that did
    # close, and say so loudly — a silent partial is indistinguishable from a page
    # that genuinely had nothing on it, which makes fill rates impossible to trust.
    print(
        f"[extractor] WARNING: unparseable JSON (finish_reason={finish_reason!r}, "
        f"{len(text)} chars). Attempting truncation repair."
    )
    for cut in range(len(text) - 1, 0, -1):
        if text[cut] != "}":
            continue
        for suffix in ("]}", "]", "}]}"):
            try:
                recovered = unwrap(json.loads(text[: cut + 1] + suffix))
            except json.JSONDecodeError:
                continue
            if recovered:
                print(f"[extractor] recovered {len(recovered)} entities from truncated JSON")
                return recovered
        break

    print("[extractor] truncation repair failed; dropping this batch")
    return None

def _build_entities(raw_list: list, plan: SearchPlan, fallback_url: str) -> list[Entity]:
    """Converts raw LLM dicts into Entity models, keeping only fields that carry a value."""
    fields = set(_data_fields(plan))
    entities: list[Entity] = []

    for item in raw_list:
        if not isinstance(item, dict):
            continue

        raw_name = item.get("name")
        if isinstance(raw_name, dict):
            raw_name = raw_name.get("value") or raw_name.get("name") or ""
        if not raw_name or not str(raw_name).strip():
            continue

        source_url = item.get("source_url") or fallback_url
        attributes: dict[str, SourcedValue] = {}

        for field, val in item.get("attributes", {}).items():
            if field not in fields:
                continue
            if isinstance(val, dict):
                value, snippet = val.get("value"), val.get("source_snippet") or ""
            else:
                value, snippet = val, ""

            # Absent means absent. Padding a miss with an explicit null is what put
            # the em-dashes in the table; the column audit decides what to do about
            # a field nothing could fill.
            if value is None or str(value).strip().lower() in (
                "", "null", "none", "n/a", "unknown", "-", "not specified", "not stated"
            ):
                continue

            attributes[field] = SourcedValue(
                value=value,
                source_url=source_url,
                source_snippet=str(snippet)[:200],
            )

        entities.append(Entity(name=str(raw_name).strip(), attributes=attributes))

    return entities

async def _extract_batch(
    pages: list[ScrapedPage],
    plan: SearchPlan,
    client: AsyncGroq,
    model: str,
    json_mode: bool,
) -> list[Entity]:
    """Runs a single extraction call over one batch of pages on one model shard."""
    if not pages:
        return []

    pages_text = "".join(_route_page(page, plan) for page in pages)
    fields_desc = ", ".join(_data_fields(plan))

    final_prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        entity_type=plan.entity_type,
        fields_desc=fields_desc,
        pages_text=pages_text,
        max_entities=ENTITIES_PER_BATCH,
    )

    degraded = False
    for attempt in range(MAX_RETRIES):
        try:
            kwargs = dict(
                model=model,
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": final_prompt},
                ],
            )
            # A prior attempt may have proved json mode unusable for this payload.
            if json_mode and not degraded:
                kwargs["response_format"] = {"type": "json_object"}
            if model.startswith("openai/gpt-oss"):
                kwargs["extra_body"] = {"reasoning_effort": REASONING_EFFORT}

            response = await client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            raw_list = _safe_parse(choice.message.content, choice.finish_reason or "")

            if raw_list is None:
                return []
            return _build_entities(raw_list, plan, pages[0].url)

        except Exception as e:
            msg = str(e)
            print(f"[extractor] {model} attempt {attempt+1}/{MAX_RETRIES}: {msg[:160]}")
            if attempt == MAX_RETRIES - 1:
                # None means the shard failed; [] would mean it found nothing.
                # Conflating the two is what turned a rate limit into a silent
                # empty table with no explanation for the user.
                return None
            # Server-side JSON validation rejects the whole call when a completion
            # truncates. Retrying in plain text lets _safe_parse repair it instead.
            if "validate JSON" in msg or "json_validate" in msg:
                degraded = True
            # Backoff stays short: the latency budget cannot absorb a 20s sleep.
            await asyncio.sleep(2 if "429" in msg else 1)

    return []

async def extract_entities(
    pages: list[ScrapedPage],
    plan: SearchPlan,
    client: AsyncGroq,
    max_pages: int = MAX_PAGES,
) -> list[Entity]:
    """Extracts entities from pages using one parallel LLM call per model shard."""
    pages = pages[:max_pages]
    if not pages:
        return []

    batches = [
        pages[i : i + PAGES_PER_BATCH] for i in range(0, len(pages), PAGES_PER_BATCH)
    ]
    batches = batches[: len(EXTRACTION_SHARDS)]

    print(f"[extractor] {len(pages)} pages -> {len(batches)} parallel calls across shards")

    tasks = {
        asyncio.create_task(
            _extract_batch(batch, plan, client, *EXTRACTION_SHARDS[i])
        ): EXTRACTION_SHARDS[i][0]
        for i, batch in enumerate(batches)
    }

    done, pending = await asyncio.wait(
        tasks.keys(), timeout=EXTRACT_DEADLINE, return_when=asyncio.FIRST_COMPLETED
    )

    # One shard is home, so there is something to show. Let the rest finish only if
    # they are close behind.
    if pending:
        extra_done, pending = await asyncio.wait(pending, timeout=EXTRACT_GRACE)
        done |= extra_done

    for task in pending:
        print(f"[extractor] deadline hit — abandoning shard {tasks[task]}")
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    entities: list[Entity] = []
    failures = len(pending)
    for task in done:
        try:
            result = task.result()
        except Exception as e:
            print(f"[extractor] shard {tasks[task]} failed: {str(e)[:120]}")
            failures += 1
            continue
        if result is None:
            failures += 1
        else:
            entities.extend(result)

    # Losing a shard to a deadline or a long reasoning excursion silently halves the
    # pages that were actually read, and used to end the query with an empty or
    # near-empty table. Top up from the pages the surviving shards never saw.
    if len(entities) < MIN_YIELD and pages:
        # If the fast shards failed rather than came up short, they are almost
        # certainly rate limited, so retrying them is pointless — go to the shard
        # on the larger bucket instead.
        model, json_mode = (
            FALLBACK_SHARD if failures >= len(batches) else EXTRACTION_SHARDS[0]
        )
        print(f"[extractor] low yield ({len(entities)}, {failures} shard failures) "
              f"— topping up on {model}")
        extra = await _extract_batch(pages[:PAGES_PER_BATCH], plan, client, model, json_mode)
        if extra is None:
            failures += 1
        else:
            seen = {e.name.lower() for e in entities}
            entities.extend(e for e in extra if e.name.lower() not in seen)
            print(f"[extractor] top-up added {len(extra)} candidates")

    if not entities and failures:
        raise ExtractionUnavailable(
            "The language model provider rejected every extraction request "
            "(usually the free-tier rate limit). Wait a minute and try again."
        )

    print(f"[extractor] {len(entities)} raw entities")
    return entities
