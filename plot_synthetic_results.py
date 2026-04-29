"""
Plots for synthetic memory evaluation results.

Tasks:
  knowledge_update  – contradiction resolution (per_gap_distance breakdown)
  slow_burn         – joint-fact retrieval (binary yes/no)
  episodic          – episodic boundary detection (per_queried_episode breakdown)

Variants: titans_original, multi_signal, dual_store
"""

import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "synthetic_tasks", "results")
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "plots")

COLORS = {
    "titans_original": "#0077b6",
    "multi_signal":    "#e63946",
    "dual_store":      "#2a9d8f",
}
LABELS = {
    "titans_original": "Titans (Original)",
    "multi_signal":    "Multi-Signal",
    "dual_store":      "Dual-Store",
}
TASK_LABELS = {
    "knowledge_update": "Knowledge Update",
    "slow_burn":        "Slow-Burn Relevance",
    "episodic":         "Episodic Boundary",
}

MARKER = {"titans_original": "o", "multi_signal": "s", "dual_store": "^"}


def _load_results():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "eval_results_*.json")))
    if not files:
        raise FileNotFoundError("No eval_results_*.json found in synthetic_tasks/results/")
    with open(files[-1]) as f:
        return json.load(f)


def _savefig(fig, name, show=False):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 1. Training loss curves – one panel per task, all variants overlaid
# ---------------------------------------------------------------------------

def plot_synthetic_train_loss(show=False):
    data = _load_results()

    tasks = ["knowledge_update", "slow_burn", "episodic"]
    variants = ["titans_original", "multi_signal", "dual_store"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)

    for ax, task in zip(axes, tasks):
        for variant in variants:
            rec = next((r for r in data if r["variant"] == variant and r["task"] == task), None)
            if rec is None:
                continue
            steps  = [m["step"]  for m in rec["train_metrics"]]
            losses = [m["loss"]  for m in rec["train_metrics"]]
            ax.plot(steps, losses,
                    label=LABELS[variant],
                    color=COLORS[variant],
                    marker=MARKER[variant], markersize=4,
                    linewidth=1.8)

        ax.set_title(TASK_LABELS[task], fontsize=11, fontweight="bold")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Synthetic Tasks – Training Loss Curves", fontsize=13, y=1.01)
    fig.tight_layout()
    return _savefig(fig, "synthetic_train_loss.png", show)


# ---------------------------------------------------------------------------
# 2. Exact-match accuracy over training checkpoints
# ---------------------------------------------------------------------------

def plot_synthetic_accuracy_curves(show=False):
    data = _load_results()

    tasks    = ["knowledge_update", "slow_burn", "episodic"]
    variants = ["titans_original", "multi_signal", "dual_store"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)

    for ax, task in zip(axes, tasks):
        for variant in variants:
            rec = next((r for r in data if r["variant"] == variant and r["task"] == task), None)
            if rec is None:
                continue
            ckpts  = rec["eval_checkpoints"]
            steps  = [c["step"]        for c in ckpts]
            accs   = [c["exact_match"] * 100 for c in ckpts]
            # also add final eval as last point
            final_step = rec["train_metrics"][-1]["step"]
            final_acc  = rec["final_eval"]["exact_match"] * 100
            if final_step not in steps:
                steps.append(final_step)
                accs.append(final_acc)

            ax.plot(steps, accs,
                    label=LABELS[variant],
                    color=COLORS[variant],
                    marker=MARKER[variant], markersize=5,
                    linewidth=1.8)

        ax.set_title(TASK_LABELS[task], fontsize=11, fontweight="bold")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Exact Match (%)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

        # chance-level line only for slow_burn (binary)
        if task == "slow_burn":
            ax.axhline(50, color="grey", linestyle="--", linewidth=1, label="Chance (50%)")
            ax.legend(fontsize=8)

    fig.suptitle("Synthetic Tasks – Exact-Match Accuracy During Training", fontsize=13, y=1.01)
    fig.tight_layout()
    return _savefig(fig, "synthetic_accuracy_curves.png", show)


# ---------------------------------------------------------------------------
# 3. Final accuracy bar chart – grouped by task
# ---------------------------------------------------------------------------

def plot_synthetic_final_accuracy(show=False):
    data = _load_results()

    tasks    = ["knowledge_update", "slow_burn", "episodic"]
    variants = ["titans_original", "multi_signal", "dual_store"]

    x       = np.arange(len(tasks))
    n_var   = len(variants)
    width   = 0.22
    offsets = np.linspace(-(n_var - 1) / 2, (n_var - 1) / 2, n_var) * width

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, variant in enumerate(variants):
        accs = []
        for task in tasks:
            rec = next((r for r in data if r["variant"] == variant and r["task"] == task), None)
            accs.append(rec["final_eval"]["exact_match"] * 100 if rec else 0)
        bars = ax.bar(x + offsets[i], accs, width,
                      label=LABELS[variant],
                      color=COLORS[variant],
                      edgecolor="white")
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in tasks], fontsize=10)
    ax.set_ylabel("Exact Match (%)")
    ax.set_title("Synthetic Tasks – Final Exact-Match Accuracy", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(
        r["final_eval"]["exact_match"] * 100
        for r in data
    ) * 1.45 + 1)

    # chance-line for slow_burn (middle bar group)
    ax.axhline(50, color="grey", linestyle="--", linewidth=0.9, label="Chance (50%)")
    ax.legend(fontsize=9)

    fig.tight_layout()
    return _savefig(fig, "synthetic_final_accuracy.png", show)


# ---------------------------------------------------------------------------
# 4. Multi-metric comparison: exact match + first_tok_acc + token_acc (final)
# ---------------------------------------------------------------------------

def plot_synthetic_metric_comparison(show=False):
    data = _load_results()

    tasks    = ["knowledge_update", "slow_burn", "episodic"]
    variants = ["titans_original", "multi_signal", "dual_store"]
    metrics  = [("exact_match", "Exact Match"), ("first_tok_acc", "First-Token Acc"), ("token_acc", "Token Acc")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        x       = np.arange(len(tasks))
        n_var   = len(variants)
        width   = 0.22
        offsets = np.linspace(-(n_var - 1) / 2, (n_var - 1) / 2, n_var) * width

        for i, variant in enumerate(variants):
            vals = []
            for task in tasks:
                rec = next((r for r in data if r["variant"] == variant and r["task"] == task), None)
                vals.append(rec["final_eval"][metric_key] * 100 if rec else 0)
            bars = ax.bar(x + offsets[i], vals, width,
                          label=LABELS[variant],
                          color=COLORS[variant],
                          edgecolor="white")
            ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([TASK_LABELS[t] for t in tasks], fontsize=8, rotation=10)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(metric_label, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Synthetic Tasks – Final Accuracy by Metric (all variants)", fontsize=13, y=1.01)
    fig.tight_layout()
    return _savefig(fig, "synthetic_metric_comparison.png", show)


# ---------------------------------------------------------------------------
# 5. Knowledge-update: accuracy by gap distance (short / medium / long)
# ---------------------------------------------------------------------------

def plot_knowledge_update_gap(show=False):
    data = _load_results()

    variants   = ["titans_original", "multi_signal", "dual_store"]
    gap_labels = ["short", "medium", "long"]

    x      = np.arange(len(gap_labels))
    width  = 0.22
    n_var  = len(variants)
    offsets = np.linspace(-(n_var - 1) / 2, (n_var - 1) / 2, n_var) * width

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, variant in enumerate(variants):
        rec = next((r for r in data if r["variant"] == variant and r["task"] == "knowledge_update"), None)
        if rec is None:
            continue
        per_gap = rec["final_eval"].get("per_gap_distance", {})
        accs = [per_gap.get(g, {}).get("acc", 0) * 100 for g in gap_labels]
        bars = ax.bar(x + offsets[i], accs, width,
                      label=LABELS[variant],
                      color=COLORS[variant],
                      edgecolor="white")
        ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(["Short Gap", "Medium Gap", "Long Gap"], fontsize=10)
    ax.set_ylabel("Exact Match (%)")
    ax.set_title("Knowledge Update – Accuracy by Gap Distance", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 3.5)

    fig.tight_layout()
    return _savefig(fig, "synthetic_knowledge_gap.png", show)


# ---------------------------------------------------------------------------
# 6. Episodic: accuracy by queried episode (ep 1 / 2 / 3)
# ---------------------------------------------------------------------------

def plot_episodic_by_episode(show=False):
    data = _load_results()

    variants = ["titans_original", "multi_signal", "dual_store"]
    ep_keys  = ["1", "2", "3"]

    x       = np.arange(len(ep_keys))
    width   = 0.22
    n_var   = len(variants)
    offsets = np.linspace(-(n_var - 1) / 2, (n_var - 1) / 2, n_var) * width

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, variant in enumerate(variants):
        rec = next((r for r in data if r["variant"] == variant and r["task"] == "episodic"), None)
        if rec is None:
            continue
        per_ep = rec["final_eval"].get("per_queried_episode", {})
        accs = [per_ep.get(ep, {}).get("acc", 0) * 100 for ep in ep_keys]
        bars = ax.bar(x + offsets[i], accs, width,
                      label=LABELS[variant],
                      color=COLORS[variant],
                      edgecolor="white")
        ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Episode {e}" for e in ep_keys], fontsize=10)
    ax.set_ylabel("Exact Match (%)")
    ax.set_title("Episodic Boundary – Accuracy by Queried Episode", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 6)

    fig.tight_layout()
    return _savefig(fig, "synthetic_episodic_by_episode.png", show)


# ---------------------------------------------------------------------------
# 7. Heatmap: exact-match accuracy across (task × variant)
# ---------------------------------------------------------------------------

def plot_accuracy_heatmap(show=False):
    data = _load_results()

    tasks    = ["knowledge_update", "slow_burn", "episodic"]
    variants = ["titans_original", "multi_signal", "dual_store"]

    matrix = np.zeros((len(variants), len(tasks)))
    for i, variant in enumerate(variants):
        for j, task in enumerate(tasks):
            rec = next((r for r in data if r["variant"] == variant and r["task"] == task), None)
            matrix[i, j] = rec["final_eval"]["exact_match"] * 100 if rec else 0

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0)

    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([TASK_LABELS[t] for t in tasks], fontsize=10)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels([LABELS[v] for v in variants], fontsize=10)

    for i in range(len(variants)):
        for j in range(len(tasks)):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.1f}%",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if val > matrix.max() * 0.6 else "black")

    plt.colorbar(im, ax=ax, label="Exact Match (%)")
    ax.set_title("Exact-Match Accuracy Heatmap (Final Eval)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _savefig(fig, "synthetic_accuracy_heatmap.png", show)


# ---------------------------------------------------------------------------
# 8. Summary dashboard: 2×3 grid combining key views
# ---------------------------------------------------------------------------

def plot_synthetic_dashboard(show=False):
    data     = _load_results()
    tasks    = ["knowledge_update", "slow_burn", "episodic"]
    variants = ["titans_original", "multi_signal", "dual_store"]

    fig = plt.figure(figsize=(18, 10))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    # --- Row 0: training loss per task ---
    for col, task in enumerate(tasks):
        ax = fig.add_subplot(gs[0, col])
        for variant in variants:
            rec = next((r for r in data if r["variant"] == variant and r["task"] == task), None)
            if rec is None:
                continue
            steps  = [m["step"]  for m in rec["train_metrics"]]
            losses = [m["loss"]  for m in rec["train_metrics"]]
            ax.plot(steps, losses, label=LABELS[variant], color=COLORS[variant],
                    linewidth=1.6, marker=MARKER[variant], markersize=3)
        ax.set_title(f"Train Loss – {TASK_LABELS[task]}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Step", fontsize=8)
        ax.set_ylabel("Loss", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
        if col == 0:
            ax.legend(fontsize=7)

    # --- Row 1: final exact-match bar per task ---
    for col, task in enumerate(tasks):
        ax = fig.add_subplot(gs[1, col])
        accs   = []
        labels = []
        colors = []
        for variant in variants:
            rec = next((r for r in data if r["variant"] == variant and r["task"] == task), None)
            accs.append(rec["final_eval"]["exact_match"] * 100 if rec else 0)
            labels.append(LABELS[variant])
            colors.append(COLORS[variant])
        bars = ax.bar(labels, accs, color=colors, edgecolor="white")
        ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=8)
        ax.set_title(f"Final Accuracy – {TASK_LABELS[task]}", fontsize=9, fontweight="bold")
        ax.set_ylabel("Exact Match (%)", fontsize=8)
        ax.tick_params(axis="x", labelsize=7, rotation=10)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", alpha=0.2)
        if task == "slow_burn":
            ax.axhline(50, color="grey", linestyle="--", linewidth=0.9)
        ymax = max(accs) if max(accs) > 0 else 1
        ax.set_ylim(0, ymax * 1.5 + 0.5)

    fig.suptitle("Synthetic Memory Evaluation – Summary Dashboard", fontsize=14, fontweight="bold", y=1.01)
    return _savefig(fig, "synthetic_dashboard.png", show)


# ---------------------------------------------------------------------------
# All-in-one
# ---------------------------------------------------------------------------

def plot_all_synthetic(show=False):
    paths = [
        plot_synthetic_train_loss(show),
        plot_synthetic_accuracy_curves(show),
        plot_synthetic_final_accuracy(show),
        plot_synthetic_metric_comparison(show),
        plot_knowledge_update_gap(show),
        plot_episodic_by_episode(show),
        plot_accuracy_heatmap(show),
        plot_synthetic_dashboard(show),
    ]
    print(f"\nAll {len(paths)} synthetic plots saved to: {PLOTS_DIR}")
    return paths


if __name__ == "__main__":
    plot_all_synthetic(show=False)