import re

from backend.models import Entity, SearchPlan, SourcedValue

# --- CONSTANTS ---
NULL_RATIO_THRESHOLD = 0.6
MIN_ENTITIES_FALLBACK = 5

def _normalize(name: str) -> str:
    """Normalizes an entity name to alphanumeric characters for accurate matching."""
    return re.sub(r"[^a-z0-9]", "", name.lower())

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
        for j, e2 in enumerate(entities):
            if j in used:
                continue
            norm_j = _normalize(e2.name)
            if norm_i in norm_j or norm_j in norm_i:
                group.append(e2)
                used.add(j)
        groups.append(group)
    return groups

def _best_value(candidates: list[SourcedValue]) -> SourcedValue:
    """Selects the longest non-null sourced value from a list of candidates."""
    non_null = [c for c in candidates if c.value is not None]
    if not non_null:
        return candidates[0]
    return max(non_null, key=lambda c: len(str(c.value)))

def _merge_group(group: list[Entity], schema_fields: list[str]) -> Entity:
    """Condenses a group of duplicate entities into a single entity containing the best attributes."""
    best = max(group, key=lambda e: e.confidence)
    merged_conf = max(e.confidence for e in group)

    attributes: dict[str, SourcedValue] = {}
    for field in schema_fields:
        candidates = []
        for e in group:
            if field in e.attributes:
                candidates.append(e.attributes[field])
        if candidates:
            attributes[field] = _best_value(candidates)
        else:
            attributes[field] = SourcedValue(value=None, source_url="", source_snippet="")

    return Entity(name=best.name, attributes=attributes, confidence=merged_conf)

async def merge_entities(entities, plan) -> list[Entity]:
    """Deduplicates raw entities and filters out sparse results."""
    if not entities:
        return []

    groups = _group_entities(entities)
    merged = [_merge_group(g, plan.schema_fields) for g in groups]
    merged = sorted(merged, key=lambda e: e.confidence, reverse=True)

    def null_ratio(e: Entity) -> float:
        """Calculates the percentage of fields that are null."""
        nulls = sum(1 for v in e.attributes.values() if v.value is None)
        return nulls / max(len(e.attributes), 1)

    filtered = [e for e in merged if null_ratio(e) < NULL_RATIO_THRESHOLD]
    return filtered if filtered else merged[:MIN_ENTITIES_FALLBACK]