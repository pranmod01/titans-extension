# Synthetic Memory Evaluation Tasks

Three structured datasets for probing failure modes of surprise-only memory gating
in small character-level models (~3–5M params, dim=256).

These are **memory-mechanism unit tests**, not NLP benchmarks. All formats use
unambiguous structured tokens so the model must learn the retrieval pattern, not
language understanding.

---

## File Structure

```
synthetic_tasks/
├── generate_tasks.py          # Main entry point
├── task1_knowledge_update.py  # Task 1 generator
├── task2_slow_burn.py         # Task 2 generator
├── task3_episodic.py          # Task 3 generator
├── vocab.py                   # Shared entity pools and filler generators
├── validate_tasks.py          # Validation and inspection script
└── data/                      # Generated JSON files (after running generate_tasks.py)
```

---

## Usage

```bash
# Generate all datasets (seed=42, 2000 train / 500 eval per task)
python generate_tasks.py

# Custom seed
python generate_tasks.py --seed 123

# Preview N examples from each task without writing files
python generate_tasks.py --preview 3

# Validate generated datasets
python validate_tasks.py
```

---

## Task 1: Knowledge Update / Contradiction Resolution

**Failure mode targeted:** Surprise-only gating stores both the original fact and
the update, with no mechanism to invalidate the old value. At query time, both
compete and the model may retrieve the stale fact.

### Format

```
FACT <attr> <entity> <val_orig> . [filler]
DISTRACTOR <attr> <other_entity> <val> . [filler]
...
UPDATE <attr> <entity> <val_new> . [filler]
QUERY <attr> <entity> ANSWER <val_new>
```

### Rules

- `FACT` introduces the initial key-value pair
- `UPDATE` overwrites the value for the **same** entity and attribute
- `DISTRACTOR` lines are facts about **other** entities — must not confuse retrieval
- 2–4 distractors per example
- `QUERY` always asks for the current (updated) value
- Gap distance between `FACT` and `UPDATE` is varied: **short** (~200 chars), **medium** (~500 chars), **long** (~800 chars)

### Statistics

| Split | n    | Gap balance           | Avg length |
|-------|------|-----------------------|------------|
| Train | 2000 | ~667 per gap category | ~927 chars |
| Eval  | 500  | ~167 per gap category | ~929 chars |

### Metric

Exact match on the answer token(s).

---

## Task 2: Slow-Burn Relevance

**Failure mode targeted:** Two individually unsurprising facts are jointly
necessary to answer a binary safety question. Surprise-only gating underweights
mundane facts with low prediction error, causing the model to miss them.

### Format

```
INFO dinner <person_A> <food> . [filler]
INFO restriction <person_B> <restriction> . [filler]
QUERY dinner-safe <person_A> <person_B> ANSWER yes|no
```

### Rules

- First `INFO` states what food person A is having for dinner
- Second `INFO` states person B's dietary restriction
- `QUERY` asks whether the dinner is safe for person B
- Answer is `yes` if the food contains none of person B's trigger allergens,
  `no` otherwise
- 50/50 yes/no balance in both splits

### Knowledge Base

**Foods (48 items):** pasta, pizza, burger, shrimp, crab, lobster, sushi, salmon,
padthai, satay, grilledchicken, salad, ricebowl, tofu, lentilsoup, ... (see `task2_slow_burn.py`)

**Restrictions (8 types):** gluten-allergy, nut-allergy, peanut-allergy, lactose,
shellfish-allergy, fish-allergy, vegetarian, vegan

### Statistics

| Split | n    | yes / no balance | Avg length |
|-------|------|------------------|------------|
| Train | 2000 | 1000 / 1000      | ~710 chars |
| Eval  | 500  | 250 / 250        | ~709 chars |

### Metric

Binary accuracy (yes/no exact match).

---

## Task 3: Episodic Boundary Detection

**Failure mode targeted:** The model blends attribute values across episode
boundaries instead of scoping retrieval to the queried episode.

### Format

```
EPISODE 1 person <name> city <c1> job <j1> pet <p1> . [filler]
EPISODE 2 person <name> city <c2> job <j2> pet <p2> . [filler]
[EPISODE 3 ... (3-episode variant only)]
QUERY EPISODE <n> <name> <attr> ANSWER <val_n>
```

### Rules

- Same person appears in every episode with **different** values for each attribute
- No attribute value is reused across episodes for the same person
- Each episode has exactly 3 attributes: city, job, pet
- 2-episode (easier) and 3-episode (harder) variants
- Query specifies which episode number to retrieve from
- Earlier episodes are harder (more interference from later overwriting)

### Category Balance (5 combos × 400 = 2000 train)

| (n_episodes, queried_ep) | Train | Eval |
|--------------------------|-------|------|
| (2, 1)                   | 400   | 100  |
| (2, 2)                   | 400   | 100  |
| (3, 1)                   | 400   | 100  |
| (3, 2)                   | 400   | 100  |
| (3, 3)                   | 400   | 100  |

### Metric

Exact match accuracy, overall and broken down by `(n_episodes, queried_ep)`.

---

## Output JSON Format

```json
{
  "task": "knowledge_update",
  "split": "train",
  "examples": [
    {
      "id": 0,
      "input_text": "FACT capital Valdoria Alphaville . ... QUERY capital Valdoria ANSWER Betaburg",
      "answer": "Betaburg",
      "metadata": {
        "gap_distance": "medium",
        "entity": "Valdoria",
        "attribute": "capital",
        "original_value": "Alphaville",
        "updated_value": "Betaburg"
      }
    }
  ]
}
```

**Invariant:** the answer token(s) are always the final tokens in `input_text`,
immediately following the literal string `ANSWER `.

---

## Filler Text

Filler is template-based text drawn from 6 topics (weather, sports, economy,
travel, science, work) with randomized slot-fill from fictional place and person
name pools. Its purpose is to:

1. Create distance between signal tokens (FACT, UPDATE, INFO, EPISODE)
2. Give the model realistic-ish text to learn language patterns from
3. Blend in with INFO/FACT lines so structured tokens do not stand out purely
   by position

Filler never contains any of the structured keywords (`FACT`, `UPDATE`,
`DISTRACTOR`, `INFO`, `QUERY`, `EPISODE`, `ANSWER`).
