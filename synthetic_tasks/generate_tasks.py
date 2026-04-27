"""Main entry point: generate all three synthetic task datasets.

Usage:
    python generate_tasks.py                 # generate all datasets with default seed
    python generate_tasks.py --seed 123      # custom seed
    python generate_tasks.py --preview 3     # print 3 examples from each task, no files
"""

import argparse
import json
import os
import sys

import task1_knowledge_update
import task2_slow_burn
import task3_episodic

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

TASK_CONFIGS = [
    {
        "key":    "knowledge_update",
        "module": task1_knowledge_update,
        "name":   "Task 1: Knowledge Update",
    },
    {
        "key":    "slow_burn",
        "module": task2_slow_burn,
        "name":   "Task 2: Slow-Burn Relevance",
    },
    {
        "key":    "episodic",
        "module": task3_episodic,
        "name":   "Task 3: Episodic Boundary Detection",
    },
]


def _write_split(task_key, split_name, examples):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{task_key}_{split_name}.json")
    payload = {
        "task":     task_key,
        "split":    split_name,
        "examples": examples,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def _print_preview(task_name, examples, n):
    import random
    sample = random.sample(examples, min(n, len(examples)))
    print(f"\n{'='*70}")
    print(f"  {task_name}  ({len(examples)} examples total)")
    print(f"{'='*70}")
    for ex in sample:
        print(f"\n  [id={ex['id']}]  answer={ex['answer']!r}")
        meta = ex.get("metadata", {})
        if meta:
            meta_str = "  meta: " + ", ".join(f"{k}={v!r}" for k, v in meta.items()
                                               if k != "episode_data")
            print(meta_str)
        text = ex["input_text"]
        # Show first 300 + last 100 chars to keep preview readable
        if len(text) > 420:
            print(f"  text: {text[:300]} ... [{len(text)} chars] ... {text[-100:]}")
        else:
            print(f"  text: {text}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic memory task datasets.")
    parser.add_argument("--seed",    type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--preview", type=int, default=0,  metavar="N",
                        help="Print N examples from each task without writing files")
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-eval",  type=int, default=500)
    args = parser.parse_args()

    for cfg in TASK_CONFIGS:
        print(f"Generating {cfg['name']} ... ", end="", flush=True)
        train, eval_ = cfg["module"].generate(
            n_train=args.n_train,
            n_eval=args.n_eval,
            seed=args.seed,
        )
        print(f"done  (train={len(train)}, eval={len(eval_)})")

        if args.preview > 0:
            _print_preview(cfg["name"], train, args.preview)
        else:
            tp = _write_split(cfg["key"], "train", train)
            ep = _write_split(cfg["key"], "eval",  eval_)
            print(f"  -> {tp}")
            print(f"  -> {ep}")

    if args.preview == 0:
        print(f"\nAll datasets written to {DATA_DIR}/")


if __name__ == "__main__":
    main()
