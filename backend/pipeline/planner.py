import json

from groq import AsyncGroq

from backend.models import SearchPlan

# --- CONSTANTS ---
# Groq retired llama-3.3-70b-versatile. gpt-oss-120b is capped at 8k TPM on the
# free tier, which is ample for a single planning call but not for extraction.
LLM_MODEL = "openai/gpt-oss-120b"
LLM_MAX_TOKENS = 1200

# Hard ceiling on how wide a table the planner may propose.
MAX_SCHEMA_FIELDS = 6

SYSTEM_PROMPT = """\
You are a research planning assistant. Given a topic query, produce a structured
search plan. Respond ONLY with a valid JSON object — no markdown fences, no preamble.
"""

USER_PROMPT = """\
Topic: "{topic}"

Produce a JSON object with these exact keys:

{{
  "entity_type": "singular noun for the kind of entity, e.g. AI healthcare startup",
  "schema_fields": ["field1", "field2", ...],
  "field_synonyms": {{
    "field1": ["synonym1", "synonym2"],
    "field2": ["synonym1", "synonym2"]
  }},
  "search_queries": ["query1", "query2", "query3", "query4"]
}}

Rules:
- schema_fields: 4–6 most useful attributes, and no more. Always start with "name".
  Choose attributes a directory or listing page would actually state for every
  entity. Prefer short factual values over descriptive ones.
  Examples — startups: name, founded, funding, focus_area, headquarters, key_product, founders
             restaurants: name, cuisine, neighborhood, price_range, notable_dish, rating
             software: name, license, language, stars, use_case, latest_version
- field_synonyms: for each schema field, list 2–4 alternative words that web pages
  commonly use to describe that concept. These will be used for content matching.
  Examples:
    "programming_language": ["tech stack", "built with", "written in", "technology"]
    "funding": ["raised", "investment", "series", "backed by", "capital"]
    "founded": ["established", "launched", "started", "incorporated", "year"]
    "use_case": ["designed for", "used for", "purpose", "solves", "helps with"]
    "license": ["open source", "MIT", "Apache", "proprietary", "free tier"]
    "rating": ["stars", "score", "reviewed", "ranked", "top rated"]
- search_queries: 4 diverse queries — include list articles, directories, news, comparisons
"""

async def plan_search(topic: str, client: AsyncGroq) -> SearchPlan:
    """Generates a structured search plan and schema from a user's natural language topic."""
    response = await client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(topic=topic)},
        ],
    )
    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]
    data = json.loads(text)
    plan = SearchPlan(**data)

    # A wide schema is the main driver of extraction failure: each extra field
    # multiplies both the JSON the model must emit and the reasoning it does first.
    # A 7-field schema measured 2.5x the output tokens of a 3-field one on the same
    # page, which is what pushed responses past the completion ceiling.
    if len(plan.schema_fields) > MAX_SCHEMA_FIELDS:
        print(f"[planner] trimming schema from {len(plan.schema_fields)} to {MAX_SCHEMA_FIELDS} fields")
        plan.schema_fields = plan.schema_fields[:MAX_SCHEMA_FIELDS]

    return plan