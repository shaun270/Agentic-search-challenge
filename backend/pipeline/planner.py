import json

from groq import AsyncGroq

from backend.models import SearchPlan

# --- CONSTANTS ---
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_MAX_TOKENS = 800

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
- schema_fields: 5–8 most useful attributes. Always start with "name".
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
    return SearchPlan(**data)