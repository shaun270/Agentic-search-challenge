import asyncio
import json
import re

from groq import AsyncGroq

from backend.models import Entity, ScrapedPage, SearchPlan, SourcedValue

# --- CONSTANTS ---
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_MAX_TOKENS = 8000
LLM_TEMPERATURE = 0.1
MAX_RETRIES = 3

SYSTEM_PROMPT = """\
You are a precise data extraction assistant. Extract structured entity data
from multiple web pages and return JSON. Only extract values directly confirmed
by the text. Do not hallucinate.
Respond ONLY with a valid JSON array — no markdown fences, no preamble.
"""

EXTRACTION_PROMPT_TEMPLATE = """\
Extract all {entity_type} entities from the focused page excerpts below.
Fields to extract: {fields_desc}

{pages_text}

CRITICAL: To save space, extract a MAXIMUM OF 10 ENTITIES total across all pages.
Return a single JSON array of ALL entities found across ALL pages:
[
  {{
    "name": "Entity Name as plain string",
    "confidence": 0.9,
    "source_url": "the page URL where this entity was found",
    "attributes": {{
      "field_name": {{
        "value": "extracted value or null",
        "source_snippet": "short verbatim phrase under 80 chars"
      }}
    }}
  }}
]

Rules:
- "name" must ALWAYS be a plain string, never an object
- Only fields from: {fields_desc}
- Set value to null if not found in the text
- Skip entities where you cannot confirm the name
- Return [] if nothing found
"""

def _route_page(page: ScrapedPage, plan: SearchPlan) -> str:
    """Filters page chunks using keyword scoring to return only highly relevant text sections."""
    if not page.chunks:
        return f"\n--- PAGE: {page.url} ---\n{page.content[:800]}\n"

    keywords = set()
    for field in plan.schema_fields:
        keywords.update(field.lower().replace("_", " ").split())
        for synonym in plan.field_synonyms.get(field, []):
            keywords.update(synonym.lower().split())
    keywords.add(plan.entity_type.lower())

    scored = []
    for sec_id, text in page.chunks.items():
        text_lower = text.lower()
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scored.append((score, sec_id, text))

    scored.sort(reverse=True)

    if scored:
        top = scored[:3]
        focused = f"\n--- PAGE: {page.url} ---\n"
        for _, sec_id, text in top:
            focused += text[:600] + "\n"
    else:
        items = list(page.chunks.values())[:2]
        focused = f"\n--- PAGE: {page.url} ---\n"
        for text in items:
            focused += text[:600] + "\n"

    return focused

def _safe_parse(text: str) -> list | None:
    """Attempts multiple extraction methods to safely parse JSON arrays from unpredictable LLM outputs."""
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()

    try:
        result = json.loads(text)
        return result if isinstance(result, list) else result.get("entities", [])
    except json.JSONDecodeError:
        pass

    for closing in ["}\n  }\n]", "    }\n]", "}]", "} ]"]:
        idx = text.rfind(closing)
        if idx != -1:
            try:
                return json.loads(text[:idx + len(closing)])
            except json.JSONDecodeError:
                continue

    # Regex fallback for heavily truncated JSON chunks
    objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
    results = []
    for obj in objects:
        try:
            parsed = json.loads(obj)
            if "name" in parsed:
                results.append(parsed)
        except json.JSONDecodeError:
            continue
    return results if results else None

async def extract_entities(
    pages: list[ScrapedPage],
    plan: SearchPlan,
    client: AsyncGroq,
    max_pages: int = 8,
) -> list[Entity]:
    """Routes scraped pages into concentrated excerpts and prompts the LLM to extract structured entities."""
    pages = pages[:max_pages]
    if not pages:
        return []

    print(f"[extractor] Routing {len(pages)} pages through Vectorless RAG...")
    routed_texts = [
        _route_page(page, plan) for page in pages
    ]
    
    pages_text = "".join([t for t in routed_texts if t])
    fields_desc = ", ".join(plan.schema_fields)
    
    final_prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        entity_type=plan.entity_type,
        fields_desc=fields_desc,
        pages_text=pages_text
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = await client.chat.completions.create(
                model=LLM_MODEL,
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": final_prompt},
                ],
            )
            text = response.choices[0].message.content.strip()
            raw_list = _safe_parse(text)

            if raw_list is None:
                print("RAW MODEL OUTPUT:", text[:300])
                print("Extraction JSON parse failed entirely")
                return []
            
            entities: list[Entity] = []
            
            for item in raw_list:
                if not isinstance(item, dict) or not item.get("name"):
                    continue

                raw_name = item["name"]
                if isinstance(raw_name, dict):
                    raw_name = raw_name.get("value") or raw_name.get("name") or ""
                if not raw_name:
                    continue

                source_url = item.get("source_url", pages[0].url if pages else "")
                attributes: dict[str, SourcedValue] = {}
                
                for field, val in item.get("attributes", {}).items():
                    if not isinstance(val, dict):
                        continue
                    attributes[field] = SourcedValue(
                        value=val.get("value"),
                        source_url=source_url,
                        source_snippet=val.get("source_snippet") or "",
                    )
                    
                for field in plan.schema_fields:
                    if field not in attributes:
                        attributes[field] = SourcedValue(
                            value=None, source_url=source_url, source_snippet=""
                        )
                        
                entities.append(Entity(
                    name=str(raw_name),
                    confidence=float(item.get("confidence", 0.8)),
                    attributes=attributes,
                ))

            print(f"[extractor] Got {len(entities)} entities in 1 LLM call")
            return entities

        except Exception as e:
            print(f"Extraction error (attempt {attempt+1}/3): {e}")
            if "429" in str(e):
                wait = 20 * (attempt + 1)
                print(f"Rate limited, waiting {wait}s...")
                await asyncio.sleep(wait)
            else:
                if attempt < 2:
                    await asyncio.sleep(2 * (attempt + 1))
                else:
                    return []

    return []