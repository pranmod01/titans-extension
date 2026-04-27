"""Validation and inspection script for synthetic task datasets.

Checks:
  1. Every example has an answer that appears in input_text at the right position
     (immediately after the last "ANSWER" token).
  2. Label balance for Task 2 (yes/no), gap balance for Task 1, episode-query
     balance for Task 3.
  3. Prints summary statistics and 3 random examples per task.

Usage:
    python validate_tasks.py
    python validate_tasks.py --data-dir path/to/data
"""

import argparse
import json
import os
import random
import collections

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

TASK_FILES = {
    "knowledge_update": ("knowledge_update_train.json", "knowledge_update_eval.json"),
    "slow_burn":        ("slow_burn_train.json",        "slow_burn_eval.json"),
    "episodic":         ("episodic_train.json",         "episodic_eval.json"),
}


# ---- per-task validators ------------------------------------------------

def _check_answer_at_end(ex):
    """Return True if 'ANSWER <answer>' appears at the very end of input_text."""
    text = ex["input_text"]
    answer = ex["answer"]
    expected_suffix = f"ANSWER {answer}"
    return text.endswith(expected_suffix)


def _check_answer_in_text(ex):
    """Return True if answer string appears somewhere in input_text."""
    return ex["answer"] in ex["input_text"]


def validate_task1(examples, tag):
    errors = []
    gap_counts = collections.Counter()
    lengths = []
    for ex in examples:
        if not _check_answer_at_end(ex):
            errors.append(f"[{tag}] id={ex['id']} answer not at end")
        # The UPDATE value must appear in text (it's the answer)
        meta = ex.get("metadata", {})
        updated = meta.get("updated_value", "")
        if updated and updated not in ex["input_text"]:
            errors.append(f"[{tag}] id={ex['id']} updated_value missing from text")
        # Original value must also appear (the FACT line)
        original = meta.get("original_value", "")
        if original and original not in ex["input_text"]:
            errors.append(f"[{tag}] id={ex['id']} original_value missing from text")
        gap_counts[meta.get("gap_distance", "?")] += 1
        lengths.append(len(ex["input_text"]))

    return errors, gap_counts, lengths


def validate_task2(examples, tag):
    errors = []
    answer_counts = collections.Counter()
    lengths = []
    for ex in examples:
        if not _check_answer_at_end(ex):
            errors.append(f"[{tag}] id={ex['id']} answer not at end")
        if ex["answer"] not in ("yes", "no"):
            errors.append(f"[{tag}] id={ex['id']} answer not yes/no: {ex['answer']!r}")
        # Verify food appears in text
        meta = ex.get("metadata", {})
        food = meta.get("food", "")
        if food and food not in ex["input_text"]:
            errors.append(f"[{tag}] id={ex['id']} food token missing from text")
        answer_counts[ex["answer"]] += 1
        lengths.append(len(ex["input_text"]))
    return errors, answer_counts, lengths


def validate_task3(examples, tag):
    errors = []
    queried_ep_counts = collections.Counter()
    n_ep_counts = collections.Counter()
    lengths = []
    for ex in examples:
        if not _check_answer_at_end(ex):
            errors.append(f"[{tag}] id={ex['id']} answer not at end")
        meta = ex.get("metadata", {})
        queried_ep = meta.get("queried_episode")
        n_ep = meta.get("n_episodes")
        queried_ep_counts[(n_ep, queried_ep)] += 1
        n_ep_counts[n_ep] += 1
        # Verify the answer value appears in the correct episode block
        ep_data = meta.get("episode_data", [])
        q_attr = meta.get("queried_attr", "")
        if ep_data and queried_ep:
            expected_val = ep_data[queried_ep - 1].get(q_attr, "")
            if expected_val != ex["answer"]:
                errors.append(
                    f"[{tag}] id={ex['id']} answer mismatch: "
                    f"expected {expected_val!r}, got {ex['answer']!r}"
                )
        lengths.append(len(ex["input_text"]))
    return errors, queried_ep_counts, n_ep_counts, lengths


# ---- printing helpers ---------------------------------------------------

def _print_random_examples(examples, n=3, title=""):
    sample = random.sample(examples, min(n, len(examples)))
    print(f"\n  --- {n} random examples from {title} ---")
    for ex in sample:
        meta = ex.get("metadata", {})
        meta_display = {k: v for k, v in meta.items() if k != "episode_data"}
        text = ex["input_text"]
        snippet = text[:280] + (f" ... [{len(text)} chars total] ... " + text[-80:] if len(text) > 380 else "")
        print(f"  id={ex['id']}  answer={ex['answer']!r}  meta={meta_display}")
        print(f"  text: {snippet}")
        print()


def _load(path):
    with open(path) as f:
        data = json.load(f)
    return data["examples"]


# ---- main ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    all_ok = True

    # ---- Task 1 ----
    print("\n" + "=" * 65)
    print("TASK 1: Knowledge Update / Contradiction Resolution")
    print("=" * 65)
    for split in ("train", "eval"):
        fname = f"knowledge_update_{split}.json"
        path = os.path.join(args.data_dir, fname)
        if not os.path.exists(path):
            print(f"  [SKIP] {path} not found")
            continue
        examples = _load(path)
        errors, gap_counts, lengths = validate_task1(examples, f"task1/{split}")
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        min_len, max_len = (min(lengths), max(lengths)) if lengths else (0, 0)
        print(f"\n  {split.upper()}  n={len(examples)}")
        print(f"  gap distribution: {dict(gap_counts)}")
        print(f"  length: avg={avg_len:.0f}  min={min_len}  max={max_len}")
        if errors:
            all_ok = False
            for e in errors[:10]:
                print(f"  ERROR: {e}")
            if len(errors) > 10:
                print(f"  ... and {len(errors)-10} more errors")
        else:
            print(f"  OK — all {len(examples)} examples passed")
    _print_random_examples(_load(os.path.join(args.data_dir, "knowledge_update_train.json"))
                           if os.path.exists(os.path.join(args.data_dir, "knowledge_update_train.json")) else [],
                           n=3, title="task1/train")

    # ---- Task 2 ----
    print("\n" + "=" * 65)
    print("TASK 2: Slow-Burn Relevance")
    print("=" * 65)
    for split in ("train", "eval"):
        fname = f"slow_burn_{split}.json"
        path = os.path.join(args.data_dir, fname)
        if not os.path.exists(path):
            print(f"  [SKIP] {path} not found")
            continue
        examples = _load(path)
        errors, answer_counts, lengths = validate_task2(examples, f"task2/{split}")
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        min_len, max_len = (min(lengths), max(lengths)) if lengths else (0, 0)
        print(f"\n  {split.upper()}  n={len(examples)}")
        print(f"  answer distribution: {dict(answer_counts)}")
        print(f"  length: avg={avg_len:.0f}  min={min_len}  max={max_len}")
        if errors:
            all_ok = False
            for e in errors[:10]:
                print(f"  ERROR: {e}")
            if len(errors) > 10:
                print(f"  ... and {len(errors)-10} more errors")
        else:
            print(f"  OK — all {len(examples)} examples passed")
    _print_random_examples(_load(os.path.join(args.data_dir, "slow_burn_train.json"))
                           if os.path.exists(os.path.join(args.data_dir, "slow_burn_train.json")) else [],
                           n=3, title="task2/train")

    # ---- Task 3 ----
    print("\n" + "=" * 65)
    print("TASK 3: Episodic Boundary Detection")
    print("=" * 65)
    for split in ("train", "eval"):
        fname = f"episodic_{split}.json"
        path = os.path.join(args.data_dir, fname)
        if not os.path.exists(path):
            print(f"  [SKIP] {path} not found")
            continue
        examples = _load(path)
        errors, queried_ep_counts, n_ep_counts, lengths = validate_task3(
            examples, f"task3/{split}"
        )
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        min_len, max_len = (min(lengths), max(lengths)) if lengths else (0, 0)
        print(f"\n  {split.upper()}  n={len(examples)}")
        print(f"  n_episodes distribution: {dict(n_ep_counts)}")
        print(f"  (n_ep, queried_ep) distribution: {dict(sorted(queried_ep_counts.items()))}")
        print(f"  length: avg={avg_len:.0f}  min={min_len}  max={max_len}")
        if errors:
            all_ok = False
            for e in errors[:10]:
                print(f"  ERROR: {e}")
            if len(errors) > 10:
                print(f"  ... and {len(errors)-10} more errors")
        else:
            print(f"  OK — all {len(examples)} examples passed")
    _print_random_examples(_load(os.path.join(args.data_dir, "episodic_train.json"))
                           if os.path.exists(os.path.join(args.data_dir, "episodic_train.json")) else [],
                           n=3, title="task3/train")

    print("\n" + "=" * 65)
    if all_ok:
        print("ALL CHECKS PASSED")
    else:
        print("VALIDATION FAILED — see errors above")
    print("=" * 65)


if __name__ == "__main__":
    main()
