"""Task 2: Slow-Burn Relevance.

Tests whether the model stores mundane facts that are individually unsurprising
but jointly important for answering a binary safety question.

Format:
  INFO dinner <person_A> <food> . [filler] INFO restriction <person_B> <restriction> .
  [filler] QUERY dinner-safe <person_A> <person_B> ANSWER yes|no
"""

import random
from vocab import generate_filler, PERSON_NAMES

# Food -> list of allergen tags present in that food
FOOD_ALLERGENS = {
    "pasta":           ["gluten"],
    "bread":           ["gluten"],
    "pizza":           ["gluten", "dairy"],
    "burger":          ["gluten", "meat"],
    "noodles":         ["gluten"],
    "croissant":       ["gluten", "dairy"],
    "waffle":          ["gluten", "dairy"],
    "pancake":         ["gluten", "dairy"],
    "cheesecake":      ["gluten", "dairy"],
    "macncheese":      ["gluten", "dairy"],
    "shrimp":          ["shellfish"],
    "crab":            ["shellfish"],
    "lobster":         ["shellfish"],
    "clam":            ["shellfish"],
    "oyster":          ["shellfish"],
    "prawn":           ["shellfish"],
    "paella":          ["shellfish", "meat"],
    "sushi":           ["fish"],
    "salmon":          ["fish"],
    "tuna":            ["fish"],
    "codfish":         ["fish"],
    "sardine":         ["fish"],
    "padthai":         ["peanuts", "gluten"],
    "satay":           ["peanuts", "meat"],
    "peanutsauce":     ["peanuts"],
    "peanutbutter":    ["peanuts"],
    "granola":         ["peanuts", "gluten"],
    "mixednuts":       ["peanuts", "nuts"],
    "almondbutter":    ["nuts"],
    "cashewcurry":     ["nuts"],
    "walnutbrownie":   ["nuts", "gluten", "dairy"],
    "grilledchicken":  ["meat"],
    "beefsteak":       ["meat"],
    "lambchop":        ["meat"],
    "porkribs":        ["meat"],
    "chickenbroth":    ["meat"],
    "veggiestir-fry":  [],
    "salad":           [],
    "fruitbowl":       [],
    "ricebowl":        [],
    "oatmeal":         ["gluten"],
    "tofu":            [],
    "tempeh":          [],
    "lentilsoup":      [],
    "chickpeacurry":   [],
    "cornchip":        [],
    "potatosoup":      [],
    "sweetpotato":     [],
    "broccolistir":    [],
    "mushroomrisotto": ["dairy"],
    "tomatorisotto":   ["dairy"],
}

# Restriction -> list of allergen tags that trigger it
RESTRICTION_TRIGGERS = {
    "gluten-allergy":     ["gluten"],
    "nut-allergy":        ["peanuts", "nuts"],
    "peanut-allergy":     ["peanuts"],
    "lactose":            ["dairy"],
    "shellfish-allergy":  ["shellfish"],
    "fish-allergy":       ["fish"],
    "vegetarian":         ["meat"],
    "vegan":              ["meat", "dairy", "fish", "shellfish"],
}

FOODS = list(FOOD_ALLERGENS.keys())
RESTRICTIONS = list(RESTRICTION_TRIGGERS.keys())


def _is_safe(food, restriction):
    triggers = RESTRICTION_TRIGGERS[restriction]
    allergens = FOOD_ALLERGENS[food]
    return not any(t in allergens for t in triggers)


def _build_example(rng, ex_id, target_answer):
    """Build one slow-burn example with the desired yes/no answer."""
    # Pick two distinct people
    person_a, person_b = rng.sample(PERSON_NAMES, 2)

    # Find a (food, restriction) pair matching target_answer
    for _ in range(200):
        food = rng.choice(FOODS)
        restriction = rng.choice(RESTRICTIONS)
        safe = _is_safe(food, restriction)
        if (safe and target_answer == "yes") or (not safe and target_answer == "no"):
            break
    else:
        raise RuntimeError("Could not find matching (food, restriction) pair")

    # Filler segments: ~200 chars each around the two INFO lines
    filler_a = generate_filler(rng, 200)
    filler_b = generate_filler(rng, 200)
    filler_c = generate_filler(rng, 80)

    info_dinner = f"INFO dinner {person_a} {food} ."
    info_restriction = f"INFO restriction {person_b} {restriction} ."
    query = f"QUERY dinner-safe {person_a} {person_b} ANSWER {target_answer}"

    input_text = " ".join([
        info_dinner,
        filler_a,
        info_restriction,
        filler_b,
        filler_c,
        query,
    ])

    return {
        "id": ex_id,
        "input_text": input_text,
        "answer": target_answer,
        "metadata": {
            "person_ordering": person_a,
            "person_restricted": person_b,
            "food": food,
            "restriction": restriction,
            "is_safe": safe,
        },
    }


def generate(n_train=2000, n_eval=500, seed=42):
    rng = random.Random(seed)

    def _make_split(n, id_offset):
        examples = []
        half = n // 2
        counts = {"yes": half, "no": n - half}
        ex_id = id_offset
        for answer, count in counts.items():
            for _ in range(count):
                examples.append(_build_example(rng, ex_id, answer))
                ex_id += 1
        rng.shuffle(examples)
        for i, ex in enumerate(examples):
            ex["id"] = id_offset + i
        return examples

    train = _make_split(n_train, 0)
    eval_ = _make_split(n_eval, n_train)
    return train, eval_
