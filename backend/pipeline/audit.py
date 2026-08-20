"""Post-extraction quality control.

Two jobs, both pure Python and both free in latency terms:

  score_confidence  attaches a confidence score to every individual cell, derived
                    from evidence rather than from the model's own opinion of itself.
  audit_columns     removes columns that too few entities could support, so the
                    table never renders a column that is mostly blank.
"""
from urllib.parse import urlparse

from backend.models import Entity

# --- CONSTANTS ---
# A column must be filled for at least this share of entities to earn its place.
MIN_FILL_RATIO = 0.5
# Never audit the table down to nothing: keep the best-filled columns even if they
# all fall below the threshold, and flag them instead of deleting them.
MIN_COLUMNS = 3
# Floor for the row/column resolution loop. Lower than MIN_COLUMNS on purpose: the
# loop must be able to keep trading columns for complete rows past the point where
# audit_columns would stop, or it exits with rows that still contain blanks — which
# is the exact failure this stage exists to prevent. Name plus one attribute is
# still a usable table; a table of em-dashes is not.
MIN_LOOP_COLUMNS = 1
# A row must carry this share of the surviving columns to be shown.
MIN_ROW_FILL = 0.6
MIN_ROWS = 3

# Confidence weights. These sum to 1.0 with the base.
BASE_CONFIDENCE = 0.35
GROUNDED_BONUS = 0.35
PARTIAL_SNIPPET_BONUS = 0.10
AGREEMENT_BONUS = 0.10
MAX_AGREEMENT_BONUS = 0.20
PROVENANCE_BONUS = 0.10

LOW_CONFIDENCE = 0.55

# Domains whose data is editorially maintained and structured.
TRUSTED_DOMAINS = (
    "wikipedia.org", "britannica.com", "crunchbase.com", "github.com",
    "michelin.com", "yelp.com", "tripadvisor.com", "eater.com",
    "bostonmagazine.com", "timeout.com", "opentable.com", "resy.com",
    ".gov", ".edu",
)

def _norm(text: str) -> str:
    """Lowercases and strips punctuation so a value can be found inside a snippet."""
    return "".join(ch for ch in str(text).lower() if ch.isalnum() or ch.isspace()).strip()

def _domain(url: str) -> str:
    """Registrable-ish domain for a URL, used for source-agreement counting."""
    try:
        return urlparse(url).netloc.replace("www.", "").lower()
    except ValueError:
        return ""

def score_confidence(entities: list[Entity], fields: list[str]) -> None:
    """Scores every cell in place from grounding, cross-source agreement and provenance.

    The model is never asked how confident it is. A model that invents a value also
    invents its own certainty, so the score is computed from three things that can
    be checked without another call:

      grounding   does the verbatim snippet actually contain the value it claims
                  to support? A fabricated value almost never survives this.
      agreement   how many independent domains reported the same value.
      provenance  whether the source is an editorially maintained site.
    """
    # value -> set of domains, per field, for cross-source agreement.
    seen: dict[str, dict[str, set[str]]] = {f: {} for f in fields}
    for entity in entities:
        for field in fields:
            if not entity.has_value(field):
                continue
            sv = entity.attributes[field]
            key = _norm(sv.value)
            seen[field].setdefault(key, set()).add(_domain(sv.source_url))

    for entity in entities:
        for field in fields:
            if not entity.has_value(field):
                continue
            sv = entity.attributes[field]
            value_norm = _norm(sv.value)
            snippet_norm = _norm(sv.source_snippet)

            score = BASE_CONFIDENCE
            reasons: list[str] = []

            if snippet_norm and value_norm and value_norm in snippet_norm:
                score += GROUNDED_BONUS
            elif snippet_norm:
                score += PARTIAL_SNIPPET_BONUS
                reasons.append("value not found verbatim in the quoted snippet")
            else:
                reasons.append("no supporting quote from the page")

            domains = seen[field].get(value_norm, set())
            sv.agreement_count = max(len(domains), 1)
            if sv.agreement_count > 1:
                score += min(
                    AGREEMENT_BONUS * (sv.agreement_count - 1), MAX_AGREEMENT_BONUS
                )
            else:
                reasons.append("reported by a single source")

            domain = _domain(sv.source_url)
            if any(t in domain for t in TRUSTED_DOMAINS):
                score += PROVENANCE_BONUS

            sv.confidence = round(min(score, 1.0), 2)
            sv.confidence_reason = "; ".join(reasons) if sv.confidence < LOW_CONFIDENCE else ""

    # A row is only as good as the cells that survived.
    for entity in entities:
        cells = [entity.attributes[f].confidence for f in fields if entity.has_value(f)]
        entity.confidence = round(sum(cells) / len(cells), 2) if cells else 0.0

def drop_empty(entities: list[Entity], fields: list[str]) -> list[Entity]:
    """Removes entities that carry no values at all.

    Must run before audit_columns. An entity with nothing in it is not a row, it is
    a name the extractor recognised and could say nothing about — and leaving it in
    poisons the per-column fill statistics: two empty rows out of three drag every
    column to 33% filled, which drops every column and trips the floor.
    """
    kept = [e for e in entities if any(e.has_value(f) for f in fields)]
    if len(kept) < len(entities):
        print(f"[audit] dropped {len(entities) - len(kept)} entities with no values")
    return kept

def audit_columns(
    entities: list[Entity], fields: list[str]
) -> tuple[list[str], list[str]]:
    """Drops columns too few entities could fill, returning (kept, dropped).

    The planner proposes a schema from the topic string alone, before any page has
    been fetched, so some columns are simply a guess that the web did not support.
    Rendering those as a column of em-dashes is what made the table look broken.
    """
    if not entities or not fields:
        return fields, []

    # Entities that carry nothing at all are not evidence that a column is
    # unfillable, only that those rows failed. Judging columns against them drags
    # every ratio down and drops columns that were doing fine.
    populated = [e for e in entities if any(e.has_value(f) for f in fields)] or entities

    fill = {
        field: sum(1 for e in populated if e.has_value(field)) / len(populated)
        for field in fields
    }

    kept = [f for f in fields if fill[f] >= MIN_FILL_RATIO]

    # Cascade guard: dropping columns can empty the table entirely, which is worse
    # than a sparse one. Below the floor, keep the best-filled columns regardless.
    if len(kept) < MIN_COLUMNS:
        best = sorted(fields, key=lambda f: fill[f], reverse=True)[:MIN_COLUMNS]
        kept = [f for f in fields if f in best]

    dropped = [f for f in fields if f not in kept]
    for field in dropped:
        print(f"[audit] dropped column '{field}' — only {fill[field]:.0%} filled")

    # The dropped columns must also leave the rows, or they resurface in the drawer
    # and the CSV export.
    for entity in entities:
        for field in dropped:
            entity.attributes.pop(field, None)

    return kept, dropped

def finalize_table(
    entities: list[Entity], fields: list[str]
) -> tuple[list[Entity], list[str], list[str]]:
    """Resolves rows and columns together so that every displayed cell is filled.

    Returns (rows, kept_fields, dropped_fields).

    A blank cell can be removed two ways: drop the row, or drop the column. Deciding
    either in isolation gives a bad table — gate the rows alone and a single stubborn
    column throws away most of the results; drop the columns alone and the table
    thins out to nothing. So the two are resolved against each other:

        while too few rows are complete, and columns remain to spare:
            drop the single least-filled column and look again

    Each removed column completes every row that was only missing that one value, so
    the loop trades the weakest column for more complete rows and stops as soon as
    there are enough. What comes out is a table where every cell in every row is
    filled, which is the guarantee the UI needs to never render a placeholder.
    """
    if not entities or not fields:
        return entities, fields, []

    fields = list(fields)
    dropped: list[str] = []

    def complete_rows(cols: list[str]) -> list[Entity]:
        """Rows carrying a value for every one of the given columns."""
        return [e for e in entities if cols and all(e.has_value(f) for f in cols)]

    while len(complete_rows(fields)) < MIN_ROWS and len(fields) > MIN_LOOP_COLUMNS:
        fill = {
            f: sum(1 for e in entities if e.has_value(f)) / len(entities) for f in fields
        }
        worst = min(fields, key=lambda f: fill[f])
        fields.remove(worst)
        dropped.append(worst)
        print(
            f"[audit] dropped column '{worst}' ({fill[worst]:.0%} filled) "
            f"to complete more rows"
        )

    rows = complete_rows(fields)

    # Last resort only: if nothing is complete even at the column floor, show the
    # fullest rows rather than an empty table. These are the only rows that can
    # still contain a gap, and the UI marks them.
    if not rows:
        rows = [e for e in entities if any(e.has_value(f) for f in fields)]
        rows.sort(key=lambda e: e.fill_ratio(fields), reverse=True)
        rows = rows[:MIN_ROWS]
        if rows:
            print("[audit] no fully complete row survived; showing the fullest available")

    for entity in entities:
        for field in dropped:
            entity.attributes.pop(field, None)

    rows.sort(key=lambda e: e.confidence, reverse=True)
    return rows, fields, dropped
