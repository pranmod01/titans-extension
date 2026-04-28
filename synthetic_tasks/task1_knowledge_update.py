"""Task 1: Knowledge Update / Contradiction Resolution.

Tests whether the model overrides old facts when a contradicting UPDATE arrives.
Surprise-only gating stores both versions with no invalidation mechanism.

Format:
  FACT <attr> <entity> <val1> . [filler] DISTRACTOR <attr> <other> <val> . [filler]
  UPDATE <attr> <entity> <val2> . [filler] QUERY <attr> <entity> ANSWER <val2>
"""

import random
from vocab import (
    COUNTRIES, CITY_VALUES, ATTRIBUTES, LEADERS,
    generate_filler,
)

# Gap distance in characters between FACT and UPDATE
GAP_TARGETS = {
    "short":  200,
    "medium": 500,
    "long":   800,
}


def _build_example(rng, ex_id, gap_cat):
    """Build one knowledge-update example."""
    gap_chars = GAP_TARGETS[gap_cat]

    # Pick entity, attribute, original value, updated value
    entity = rng.choice(COUNTRIES)
    attribute = rng.choice(ATTRIBUTES)

    # Value pool depends on attribute
    if attribute in ("capital",):
        pool = CITY_VALUES
    elif attribute in ("leader", "anthem", "symbol", "flag", "currency", "language"):
        pool = CITY_VALUES  # reuse for simplicity; all single-token fictional names

    val_orig, val_update = rng.sample(pool, 2)

    # 2-4 distractor facts about OTHER entities
    n_distractors = rng.randint(2, 4)
    distractor_entities = rng.sample([c for c in COUNTRIES if c != entity], n_distractors)
    distractor_vals = rng.choices(pool, k=n_distractors)
    distractor_attrs = rng.choices(ATTRIBUTES, k=n_distractors)

    # --- Build text segments ---
    fact_line = f"FACT {attribute} {entity} {val_orig} ."

    # Distribute gap_chars across segments between FACT and UPDATE
    n_segments = n_distractors + 1  # gaps between FACT...distractor(s)...UPDATE
    per_segment = max(40, gap_chars // n_segments)

    parts = [fact_line]

    for i, (d_ent, d_val, d_attr) in enumerate(
        zip(distractor_entities, distractor_vals, distractor_attrs)
    ):
        parts.append(generate_filler(rng, per_segment))
        parts.append(f"DISTRACTOR {d_attr} {d_ent} {d_val} .")

    parts.append(generate_filler(rng, per_segment))
    parts.append(f"UPDATE {attribute} {entity} {val_update} .")

    # Small filler after UPDATE before QUERY
    parts.append(generate_filler(rng, 60))
    parts.append(f"QUERY {attribute} {entity} ANSWER {val_update}")

    input_text = " ".join(parts)

    return {
        "id": ex_id,
        "input_text": input_text,
        "answer": val_update,
        "metadata": {
            "gap_distance": gap_cat,
            "entity": entity,
            "attribute": attribute,
            "original_value": val_orig,
            "updated_value": val_update,
        },
    }


def generate(n_train=2000, n_eval=500, seed=42):
    rng = random.Random(seed)
    gap_cats = list(GAP_TARGETS.keys())

    def _make_split(n, id_offset):
        examples = []
        per_cat = n // len(gap_cats)
        remainder = n % len(gap_cats)
        counts = {c: per_cat for c in gap_cats}
        for i, c in enumerate(gap_cats):
            if i < remainder:
                counts[c] += 1
        ex_id = id_offset
        for gap_cat, count in counts.items():
            for _ in range(count):
                examples.append(_build_example(rng, ex_id, gap_cat))
                ex_id += 1
        rng.shuffle(examples)
        # Re-assign ids after shuffle
        for i, ex in enumerate(examples):
            ex["id"] = id_offset + i
        return examples

    train = _make_split(n_train, 0)
    eval_ = _make_split(n_eval, n_train)
    return train, eval_
