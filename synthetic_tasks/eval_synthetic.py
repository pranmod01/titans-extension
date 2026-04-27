"""
Evaluate three memory architecture variants on the three synthetic tasks.

Models compared:
  titans_original  -- OG Titans: surprise-only gating, single NeuralMemory
  multi_signal     -- Composite G_t (S+R+C), single store
  dual_store       -- Composite G_t + fast/slow stores with consolidation

Usage (from the repo root, with the venv active):
  python synthetic_tasks/eval_synthetic.py                 # all 3 tasks
  python synthetic_tasks/eval_synthetic.py --task 1        # task 1 only
  python synthetic_tasks/eval_synthetic.py --max_steps 500 --batch_size 8
  python synthetic_tasks/eval_synthetic.py --eval_only --checkpoint_dir ./syn_ckpts
  python synthetic_tasks/eval_synthetic.py --dry_run       # 50 examples, smoke-test

Outputs:
  synthetic_tasks/results/eval_results_<timestamp>.json
  synthetic_tasks/results/eval_summary_<timestamp>.txt
"""

import os
import sys
import json
import math
import time
import random
import argparse
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup: allow running from repo root OR from synthetic_tasks/
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from multi_signal_titans.config import Config, default_config
from multi_signal_titans.transformer import create_model, count_parameters


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR   = _HERE / "data"
RESULT_DIR = _HERE / "results"

# Character-level vocab: printable ASCII (32-126) + a few extras
# The tokeniser maps each byte in the UTF-8 encoded input_text to an id.
NUM_TOKENS = 256  # byte-level, same as the rest of the codebase

# Training hypers (fine-tuning style — small task-specific datasets)
DEFAULT_MAX_STEPS   = 2000
DEFAULT_BATCH_SIZE  = 4
DEFAULT_SEQ_LEN     = 1024
DEFAULT_LR          = 3e-4
DEFAULT_WARMUP      = 200
DEFAULT_LOG_EVERY   = 100
DEFAULT_EVAL_EVERY  = 200
DEFAULT_GRAD_CLIP   = 1.0
SEED                = 42


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SyntheticTaskDataset(Dataset):
    """
    Each example is a fixed-length byte-level sequence.

    The input_text is encoded as UTF-8 bytes, truncated or padded to seq_len+1.

    The answer always appears at the end of the *original* (unpadded) text,
    immediately after "ANSWER ".  We store the exact byte range [ans_start,
    ans_end) within the padded sequence so the evaluator knows which positions
    to score — even when the text is shorter than seq_len.

    At inference time we feed inp = token_ids[:-1] (seq_len tokens) and compare
    greedy argmax at positions [ans_start-1 .. ans_end-1] in the output against
    the ground-truth answer bytes.  (The -1 shift is because the model predicts
    token t+1 from position t.)
    """

    def __init__(
        self,
        examples: List[Dict],
        seq_len: int = DEFAULT_SEQ_LEN,
        pad_id: int = 0,
    ):
        self.seq_len  = seq_len
        self.pad_id   = pad_id
        self.records  = []

        for ex in examples:
            text   = ex["input_text"]
            answer = ex["answer"]
            meta   = ex.get("metadata", {})

            # Encode to bytes
            enc    = list(text.encode("utf-8"))
            ans_b  = list(answer.encode("utf-8"))
            ans_len = len(ans_b)

            # The answer is guaranteed to be the final token(s) of input_text.
            # Find the byte range within enc.
            raw_len   = len(enc)
            ans_end   = raw_len          # exclusive, within enc
            ans_start = raw_len - ans_len  # inclusive

            # Pad/truncate to seq_len+1
            if raw_len < seq_len + 1:
                enc = enc + [pad_id] * (seq_len + 1 - raw_len)
            else:
                enc = enc[:seq_len + 1]
                # If we truncated past the answer, clamp (shouldn't happen for
                # well-formed examples where answer is at the end)
                ans_end   = min(ans_end,   seq_len + 1)
                ans_start = min(ans_start, ans_end - ans_len)

            token_ids = torch.tensor(enc, dtype=torch.long)

            self.records.append({
                "token_ids":  token_ids,    # [seq_len+1]
                "ans_bytes":  ans_b,         # list of ints (ground-truth)
                "ans_len":    ans_len,
                "ans_start":  ans_start,     # byte index in token_ids where answer begins
                "ans_end":    ans_end,        # exclusive
                "answer_str": answer,
                "id":         ex.get("id", -1),
                "meta":       meta,
            })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return r["token_ids"]   # collate returns [B, seq_len+1]


def _load_json(path: Path) -> List[Dict]:
    with open(path) as f:
        data = json.load(f)
    return data["examples"]


def build_loaders(
    task_key: str,
    seq_len: int,
    batch_size: int,
    dry_run: bool = False,
) -> Tuple["SyntheticTaskDataset", "SyntheticTaskDataset", DataLoader, DataLoader]:
    train_path = DATA_DIR / f"{task_key}_train.json"
    eval_path  = DATA_DIR / f"{task_key}_eval.json"

    train_examples = _load_json(train_path)
    eval_examples  = _load_json(eval_path)

    if dry_run:
        train_examples = train_examples[:50]
        eval_examples  = eval_examples[:50]

    train_ds = SyntheticTaskDataset(train_examples, seq_len)
    eval_ds  = SyntheticTaskDataset(eval_examples,  seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=False
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False
    )
    return train_ds, eval_ds, train_loader, eval_loader


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

VARIANTS = [
    ("titans_original", "OG Titans (surprise-only, single store)"),
    ("multi_signal",    "Multi-signal single-store"),
    ("dual_store",      "Multi-signal dual-store"),
]


def build_config(seq_len: int) -> Config:
    cfg = deepcopy(default_config)
    cfg.model.num_tokens = NUM_TOKENS
    cfg.model.dim        = 256
    cfg.model.depth      = 2
    cfg.model.heads      = 4
    cfg.model.dim_head   = 64
    cfg.model.segment_len = min(128, seq_len // 4)
    cfg.training.use_amp  = False  # NeuralMemory uses torch.func.grad; AMP breaks it
    return cfg


def make_model(variant: str, cfg: Config, device: torch.device) -> nn.Module:
    model = create_model(variant, cfg)
    model = model.to(device)
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _warmup_lr(step: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    return 1.0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    step_offset: int,
    max_steps: int,
    log_every: int,
    variant_name: str,
) -> Tuple[int, List[Dict]]:
    """Train for one pass through the loader or until max_steps. Returns (step, metrics)."""
    model.train()
    metrics = []
    step = step_offset
    acc_loss = 0.0
    acc_n    = 0

    it = loader if not _HAS_TQDM else tqdm(loader, desc=f"  {variant_name} train", leave=False)
    for batch in it:
        if step >= max_steps:
            break

        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward: shift input/target inside model when return_loss=True
        try:
            loss, _, _ = model(batch, return_loss=True, return_metrics=False)
        except TypeError:
            loss = model(batch, return_loss=True)

        if torch.isnan(loss):
            print(f"    [WARN] NaN loss at step {step}, skipping batch")
            step += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULT_GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        acc_loss += loss.item()
        acc_n    += 1
        step     += 1

        if step % log_every == 0:
            avg = acc_loss / acc_n
            metrics.append({"step": step, "loss": avg})
            if _HAS_TQDM:
                it.set_postfix(loss=f"{avg:.4f}")
            acc_loss = 0.0
            acc_n    = 0

    return step, metrics


# ---------------------------------------------------------------------------
# Evaluation: answer-level exact match
# ---------------------------------------------------------------------------

def evaluate_exact_match(
    model: nn.Module,
    eval_ds: "SyntheticTaskDataset",
    eval_loader: DataLoader,
    device: torch.device,
    seq_len: int,
    per_meta_key: Optional[str] = None,
) -> Dict:
    """
    Compute exact-match accuracy on the answer tokens.

    The model receives inp = token_ids[:-1] (seq_len bytes) and predicts
    token_ids[1:].  The answer occupies bytes [ans_start, ans_end) of
    token_ids, so in the model's output it corresponds to positions
    [ans_start-1, ans_end-1) (next-token prediction shift).

    per_meta_key: if set, also break down accuracy by that metadata field.
    """
    model.eval()

    correct   = 0
    total     = 0
    per_meta  = {}  # meta_val -> (correct, total)

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            eval_loader if not _HAS_TQDM
            else tqdm(eval_loader, desc="  eval", leave=False)
        ):
            batch = batch.to(device)   # [B, seq_len+1]
            B = batch.shape[0]

            inp = batch[:, :-1]        # [B, seq_len]

            try:
                logits, _, _ = model(inp, return_metrics=False)
            except TypeError:
                logits = model(inp)

            # logits: [B, seq_len, vocab]
            preds = logits.argmax(dim=-1)   # [B, seq_len]

            start = batch_idx * eval_loader.batch_size
            for i in range(B):
                ex_idx = start + i
                if ex_idx >= len(eval_ds.records):
                    break
                rec = eval_ds.records[ex_idx]
                ans_b     = rec["ans_bytes"]
                ans_start = rec["ans_start"]   # byte index in token_ids
                ans_end   = rec["ans_end"]

                # In the prediction sequence (output of inp), position p
                # predicts token_ids[p+1], so the answer at [ans_start, ans_end)
                # in token_ids is predicted by output positions [ans_start-1, ans_end-1).
                pred_start = ans_start - 1
                pred_end   = ans_end   - 1

                if pred_start < 0 or pred_end > seq_len:
                    # Degenerate: answer outside the sequence window; skip
                    total += 1
                    if per_meta_key:
                        mval = str(rec["meta"].get(per_meta_key, "unknown"))
                        c, t = per_meta.get(mval, (0, 0))
                        per_meta[mval] = (c, t + 1)
                    continue

                pred_positions = preds[i, pred_start:pred_end]   # [ans_len]
                gold = torch.tensor(ans_b, device=device)

                is_correct = bool((pred_positions == gold).all().item())
                correct += int(is_correct)
                total   += 1

                if per_meta_key:
                    mval = str(rec["meta"].get(per_meta_key, "unknown"))
                    c, t = per_meta.get(mval, (0, 0))
                    per_meta[mval] = (c + int(is_correct), t + 1)

    result = {
        "exact_match": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
    }
    if per_meta_key and per_meta:
        result["per_" + per_meta_key] = {
            k: {"acc": c / t, "n": t} for k, (c, t) in sorted(per_meta.items())
        }
    return result


# ---------------------------------------------------------------------------
# Per-task breakdown keys
# ---------------------------------------------------------------------------

TASK_META_KEY = {
    "knowledge_update": "gap_distance",
    "slow_burn":        None,           # binary — no breakdown needed
    "episodic":         "queried_episode",
}


# ---------------------------------------------------------------------------
# Main training+eval loop for one variant × one task
# ---------------------------------------------------------------------------

def run_variant_on_task(
    variant: str,
    task_key: str,
    cfg: Config,
    device: torch.device,
    max_steps: int,
    batch_size: int,
    seq_len: int,
    log_every: int,
    eval_every: int,
    dry_run: bool,
    checkpoint_dir: Optional[Path],
    eval_only: bool,
) -> Dict:
    t0 = time.time()

    # ---- data ----
    train_ds, eval_ds, train_loader, eval_loader = build_loaders(
        task_key, seq_len, batch_size, dry_run
    )

    # ---- model ----
    model = make_model(variant, cfg, device)
    n_params = count_parameters(model)

    ckpt_path = None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoint_dir / f"{variant}_{task_key}.pt"

    start_step = 0
    if eval_only and ckpt_path and ckpt_path.exists():
        print(f"    Loading checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        start_step = state.get("step", max_steps)
    elif ckpt_path and ckpt_path.exists() and not eval_only:
        print(f"    Resuming from checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        start_step = state.get("step", 0)

    # ---- train ----
    train_metrics = []
    eval_checkpoints = []

    if not eval_only and start_step < max_steps:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=DEFAULT_LR,
            weight_decay=1e-2,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda s: _warmup_lr(s + start_step, DEFAULT_WARMUP)
        )

        step = start_step
        epoch = 0
        while step < max_steps:
            step, new_metrics = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                device, step, max_steps, log_every, variant,
            )
            train_metrics.extend(new_metrics)
            epoch += 1

            # Mid-training eval checkpoint
            if step % eval_every == 0 and step < max_steps:
                mid_res = evaluate_exact_match(
                    model, eval_ds, eval_loader, device, seq_len,
                    per_meta_key=TASK_META_KEY.get(task_key),
                )
                eval_checkpoints.append({"step": step, **mid_res})
                print(f"    step={step:5d}  eval_acc={mid_res['exact_match']:.3f}")

        # Save checkpoint
        if ckpt_path:
            torch.save({"model": model.state_dict(), "step": step}, ckpt_path)

    # ---- final eval ----
    meta_key = TASK_META_KEY.get(task_key)
    final_eval = evaluate_exact_match(
        model, eval_ds, eval_loader, device, seq_len,
        per_meta_key=meta_key,
    )

    elapsed = time.time() - t0
    return {
        "variant":        variant,
        "task":           task_key,
        "n_params":       n_params,
        "train_metrics":  train_metrics,
        "eval_checkpoints": eval_checkpoints,
        "final_eval":     final_eval,
        "elapsed_s":      elapsed,
    }


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

TASK_LABELS = {
    "knowledge_update": "Task 1: Knowledge Update",
    "slow_burn":        "Task 2: Slow-Burn Relevance",
    "episodic":         "Task 3: Episodic Boundary",
}

VARIANT_LABELS = {
    "titans_original": "OG Titans       ",
    "multi_signal":    "Multi-sig single",
    "dual_store":      "Multi-sig dual  ",
}


def print_summary(all_results: List[Dict]):
    tasks = sorted({r["task"] for r in all_results})
    variants = [v for v, _ in VARIANTS if any(r["variant"] == v for r in all_results)]

    print("\n" + "=" * 72)
    print("SYNTHETIC TASK EVALUATION SUMMARY")
    print("=" * 72)

    for task in tasks:
        print(f"\n  {TASK_LABELS.get(task, task)}")
        print(f"  {'Variant':<24} {'Acc':>6}  {'N':>5}  {'Params':>9}  {'Time':>7}")
        print(f"  {'-'*24} {'-'*6}  {'-'*5}  {'-'*9}  {'-'*7}")
        for v in variants:
            matches = [r for r in all_results if r["task"] == task and r["variant"] == v]
            if not matches:
                continue
            r = matches[0]
            fe = r["final_eval"]
            acc    = fe.get("exact_match", 0.0)
            n      = fe.get("total", 0)
            params = r.get("n_params", 0)
            t      = r.get("elapsed_s", 0)
            print(f"  {VARIANT_LABELS.get(v, v):<24} {acc:6.3f}  {n:>5}  {params:>9,}  {t:6.0f}s")

            # Per-meta breakdown
            for mk in ["per_gap_distance", "per_queried_episode"]:
                if mk in fe:
                    for mval, stats in sorted(fe[mk].items()):
                        print(f"    {'':<20} {mval:<10}: {stats['acc']:.3f}  (n={stats['n']})")

    print("\n" + "=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate memory architectures on synthetic tasks")
    parser.add_argument("--task", type=int, choices=[1, 2, 3], default=None,
                        help="Run only task 1/2/3 (default: all)")
    parser.add_argument("--variant", choices=[v for v, _ in VARIANTS], default=None,
                        help="Run only one model variant (default: all three)")
    parser.add_argument("--max_steps",  type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seq_len",    type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--log_every",  type=int, default=DEFAULT_LOG_EVERY)
    parser.add_argument("--eval_every", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--seed",       type=int, default=SEED)
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory for saving/loading model checkpoints")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, load checkpoints and evaluate only")
    parser.add_argument("--dry_run", action="store_true",
                        help="50 examples per task — quick smoke test (no GPU needed)")
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    # --- seeds ---
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # --- device ---
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(f"Device: {device}")

    # --- task / variant selection ---
    task_map = {
        1: "knowledge_update",
        2: "slow_burn",
        3: "episodic",
    }
    if args.task is not None:
        tasks_to_run = [task_map[args.task]]
    else:
        tasks_to_run = list(task_map.values())

    if args.variant is not None:
        variants_to_run = [(args.variant, dict(VARIANTS)[args.variant])]
    else:
        variants_to_run = VARIANTS

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None

    # --- shared config (same for all variants) ---
    cfg = build_config(args.seq_len)

    if args.dry_run:
        args.max_steps  = 50
        args.eval_every = 25
        args.log_every  = 10
        print("DRY RUN: 50 examples, 50 steps per variant×task")

    # --- run ---
    all_results = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for task_key in tasks_to_run:
        print(f"\n{'='*60}")
        print(f"  {TASK_LABELS.get(task_key, task_key)}")
        print(f"{'='*60}")

        for variant, variant_label in variants_to_run:
            print(f"\n  Variant: {variant_label}")
            torch.manual_seed(args.seed)   # reset per variant for fair comparison

            try:
                result = run_variant_on_task(
                    variant       = variant,
                    task_key      = task_key,
                    cfg           = build_config(args.seq_len),  # fresh config per run
                    device        = device,
                    max_steps     = args.max_steps,
                    batch_size    = args.batch_size,
                    seq_len       = args.seq_len,
                    log_every     = args.log_every,
                    eval_every    = args.eval_every,
                    dry_run       = args.dry_run,
                    checkpoint_dir= ckpt_dir,
                    eval_only     = args.eval_only,
                )
                fe = result["final_eval"]
                print(f"  -> final eval acc: {fe['exact_match']:.4f}  "
                      f"({fe['correct']}/{fe['total']})  "
                      f"elapsed: {result['elapsed_s']:.0f}s")
            except Exception:
                print(f"  [ERROR] {variant} on {task_key} failed:")
                traceback.print_exc()
                result = {
                    "variant": variant, "task": task_key,
                    "error": traceback.format_exc(),
                    "final_eval": {"exact_match": 0.0, "correct": 0, "total": 0},
                }

            all_results.append(result)

    # --- save results ---
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    results_path = RESULT_DIR / f"eval_results_{ts}.json"
    summary_path = RESULT_DIR / f"eval_summary_{ts}.txt"

    # JSON: strip tensors (should be none, but safety)
    def _serialise(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _serialise(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialise(v) for v in obj]
        return obj

    with open(results_path, "w") as f:
        json.dump(_serialise(all_results), f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Human-readable summary
    print_summary(all_results)

    import io
    buf = io.StringIO()
    _orig_stdout = sys.stdout
    sys.stdout = buf
    print_summary(all_results)
    sys.stdout = _orig_stdout
    with open(summary_path, "w") as f:
        f.write(buf.getvalue())
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
