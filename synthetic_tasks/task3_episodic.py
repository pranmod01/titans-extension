"""Task 3: Episodic Boundary Detection.

Tests whether the model scopes memory to the correct episode instead of blending
facts across episode boundaries.

Format (2-episode):
  EPISODE 1 person John city Paris job baker pet cat . [filler]
  EPISODE 2 person John city Tokyo job banker pet dog . [filler]
  QUERY EPISODE 1 John city ANSWER Paris

Format (3-episode):
  EPISODE 1 ... EPISODE 2 ... EPISODE 3 ...
  QUERY EPISODE <n> <person> <attr> ANSWER <val>
"""

import random
from vocab import PERSON_NAMES, JOBS, PETS, CITIES, generate_filler

# Attribute pool per slot — single-token values only
ATTR_CITY_POOL = [c for c in CITIES if " " not in c]
ATTR_JOB_POOL  = [j for j in JOBS  if " " not in j]
ATTR_PET_POOL  = [p for p in PETS  if " " not in p]

EPISODE_ATTRIBUTES = ["city", "job", "pet"]
ATTR_POOLS = {
    "city": ATTR_CITY_POOL,
    "job":  ATTR_JOB_POOL,
    "pet":  ATTR_PET_POOL,
}


def _build_episode_block(rng, ep_num, person, attr_values):
    """Return the EPISODE line string for one episode.

    attr_values: dict {attr: value}
    """
    attrs_str = " ".join(
        f"{attr} {val}" for attr, val in attr_values.items()
    )
    return f"EPISODE {ep_num} person {person} {attrs_str} ."


def _build_example(rng, ex_id, n_episodes, queried_ep):
    """Build one episodic boundary example.

    n_episodes: 2 or 3
    queried_ep: which episode number is queried (1-indexed)
    """
    person = rng.choice(PERSON_NAMES)
    queried_attr = rng.choice(EPISODE_ATTRIBUTES)

    # Assign distinct values per episode for each attribute
    episode_data = []  # list of dicts {attr: value}
    for _ in range(n_episodes):
        ep_attrs = {}
        for attr in EPISODE_ATTRIBUTES:
            ep_attrs[attr] = None
        episode_data.append(ep_attrs)

    # For each attribute, sample n_episodes distinct values
    for attr in EPISODE_ATTRIBUTES:
        pool = ATTR_POOLS[attr]
        chosen = rng.sample(pool, n_episodes)
        for ep_idx in range(n_episodes):
            episode_data[ep_idx][attr] = chosen[ep_idx]

    correct_answer = episode_data[queried_ep - 1][queried_attr]

    # Build the full text
    filler_per_ep = 200  # chars of filler after each episode block
    parts = []
    for ep_num in range(1, n_episodes + 1):
        block = _build_episode_block(rng, ep_num, person, episode_data[ep_num - 1])
        parts.append(block)
        parts.append(generate_filler(rng, filler_per_ep))

    # Small trailing filler before QUERY
    parts.append(generate_filler(rng, 60))
    parts.append(
        f"QUERY EPISODE {queried_ep} {person} {queried_attr} ANSWER {correct_answer}"
    )

    input_text = " ".join(parts)

    return {
        "id": ex_id,
        "input_text": input_text,
        "answer": correct_answer,
        "metadata": {
            "n_episodes": n_episodes,
            "queried_episode": queried_ep,
            "person": person,
            "queried_attr": queried_attr,
            "episode_data": episode_data,
        },
    }


def generate(n_train=2000, n_eval=500, seed=42):
    rng = random.Random(seed)

    # Build category breakdown: (n_episodes, queried_ep) combos
    # 2-ep: query ep1, ep2        → 2 combos
    # 3-ep: query ep1, ep2, ep3   → 3 combos
    # Total 5 combos — balance as evenly as possible
    categories = [
        (2, 1), (2, 2),
        (3, 1), (3, 2), (3, 3),
    ]

    def _make_split(n, id_offset):
        examples = []
        per_cat = n // len(categories)
        remainder = n % len(categories)
        counts = {c: per_cat for c in categories}
        for i, c in enumerate(categories):
            if i < remainder:
                counts[c] += 1

        ex_id = id_offset
        for (n_ep, q_ep), count in counts.items():
            for _ in range(count):
                examples.append(_build_example(rng, ex_id, n_ep, q_ep))
                ex_id += 1

        rng.shuffle(examples)
        for i, ex in enumerate(examples):
            ex["id"] = id_offset + i
        return examples

    train = _make_split(n_train, 0)
    eval_ = _make_split(n_eval, n_train)
    return train, eval_
