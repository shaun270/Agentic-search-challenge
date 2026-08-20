import re

from backend.models import Entity, SearchPlan, SourcedValue

# --- CONSTANTS ---
# Two names are the same entity if one contains the other AND they are close in
# length. Bare containment collapsed unrelated rows together: "Ora" is a substring
# of "Corallo", which merged 16 extracted entities down to 1.
NAME_LENGTH_RATIO = 0.7

def _normalize(name: str) -> str:
    """Normalizes an entity name to alphanumeric characters for accurate matching."""
    return re.sub(r"[^a-z0-9]", "", name.lower())

def _same_entity(a: str, b: str) -> bool:
    """True when two normalized names plausibly denote the same entity."""
    if not a or not b:
        return False
    if a == b:
        return True
    if a in b or b in a:
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        return len(shorter) / len(longer) >= NAME_LENGTH_RATIO
    return False

def _group_entities(entities: list[Entity]) -> list[list[Entity]]:
    """Groups identically or similarly named entities together."""
    groups: list[list[Entity]] = []
    used = set()
    for i, e in enumerate(entities):
        if i in used:
            continue
        group = [e]
        used.add(i)
        norm_i = _normalize(e.name)
        for j in range(i + 1, len(entities)):
            if j in used:
                continue
            if _same_entity(norm_i, _normalize(entities[j].name)):
                group.append(entities[j])
                used.add(j)
        groups.append(group)
    return groups

def _best_value(candidates: list[SourcedValue]) -> SourcedValue:
    """Selects the best-evidenced value from a list of candidates.

    Previously this returned the longest string, on the theory that longer meant
    more informative. In practice it meant a hedging sentence beat the real answer:
    "prices vary depending on the season" outranked "$$".
    """
    grounded = [
        c for c in candidates
        if c.source_snippet and str(c.value).lower() in c.source_snippet.lower()
    ]
    pool = grounded or candidates
    # Prefer a value backed by a quote; break ties toward the more concise answer,
    # which for an attribute is almost always the more precise one.
    return min(pool, key=lambda c: (not c.source_snippet, len(str(c.value))))

def _merge_group(group: list[Entity], schema_fields: list[str]) -> Entity:
    """Condenses a group of duplicate entities into a single entity containing the best attributes."""
    best = max(group, key=lambda e: len(e.attributes))

    attributes: dict[str, SourcedValue] = {}
    for field in schema_fields:
        candidates = [
            e.attributes[field] for e in group
            if field in e.attributes and e.has_value(field)
        ]
        # A field nothing could fill is simply absent. Writing an explicit null here
        # is what produced the column of em-dashes; the column audit decides what to
        # do about a field the sources could not support.
        if candidates:
            attributes[field] = _best_value(candidates)

    return Entity(name=best.name, attributes=attributes, confidence=0.0)

async def merge_entities(entities, plan) -> list[Entity]:
    """Deduplicates raw entities across shards without discarding sparse ones.

    Filtering happens later, in the column audit, which can only judge a row once
    it knows which columns survived.
    """
    if not entities:
        return []

    fields = [f for f in plan.schema_fields if f.lower() != "name"]
    groups = _group_entities(entities)
    merged = [_merge_group(g, fields) for g in groups]

    print(f"[merger] {len(entities)} raw -> {len(merged)} unique entities")
    return merged
