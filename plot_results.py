"""
Plotting functions for titans-extension experiment results.
Each function loads the latest run for a given experiment number.
"""

import json
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "experiment_results")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")

COLORS = {
    "vanilla": "#6c757d",
    "Vanilla Transformer": "#6c757d",
    "titans_original": "#0077b6",
    "multi_signal": "#e63946",
    "dual_store": "#2a9d8f",
    "surprise_only": "#457b9d",
    "surprise_relevance": "#e9c46a",
    "surprise_contiguity": "#f4a261",
    "full_composite": "#e63946",
}

LABELS = {
    "vanilla": "Vanilla Transformer",
    "Vanilla Transformer": "Vanilla Transformer",
    "titans_original": "Titans (Original)",
    "multi_signal": "Multi-Signal",
    "dual_store": "Dual-Store",
    "surprise_only": "Surprise Only",
    "surprise_relevance": "Surprise + Relevance",
    "surprise_contiguity": "Surprise + Contiguity",
    "full_composite": "Full Composite",
}


def _latest_file(exp_num: int) -> str:
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"experiment_{exp_num}_*.json")))
    if not files:
        raise FileNotFoundError(f"No results found for experiment {exp_num}")
    return files[-1]


def _load(exp_num: int) -> dict:
    with open(_latest_file(exp_num)) as f:
        return json.load(f)


def _savefig(fig: plt.Figure, name: str, show: bool) -> str:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Experiment 1 – training loss curves: Vanilla vs Titans-original vs Multi-signal
# ---------------------------------------------------------------------------

def plot_exp1_loss_curves(show: bool = True) -> str:
    """Line plot of training loss over steps for all three models."""
    data = _load(1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for model_key, records in data.items():
        steps = [r["step"] for r in records]
        losses = [r["loss"] for r in records]
        ax.plot(steps, losses,
                label=LABELS.get(model_key, model_key),
                color=COLORS.get(model_key, None),
                linewidth=2)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Exp 1 – Training Loss: Vanilla vs Titans vs Multi-Signal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _savefig(fig, "exp1_loss_curves.png", show)


def plot_exp1_final_loss_bar(show: bool = True) -> str:
    """Bar chart comparing final-step loss across models."""
    data = _load(1)

    model_keys = list(data.keys())
    final_losses = [data[k][-1]["loss"] for k in model_keys]
    labels = [LABELS.get(k, k) for k in model_keys]
    colors = [COLORS.get(k, "#888") for k in model_keys]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, final_losses, color=colors, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylabel("Final Loss")
    ax.set_title("Exp 1 – Final Training Loss Comparison")
    ax.set_ylim(0, max(final_losses) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _savefig(fig, "exp1_final_loss_bar.png", show)


# ---------------------------------------------------------------------------
# Experiment 2 – signal ablation: loss curves + gate weights evolution
# ---------------------------------------------------------------------------

def plot_exp2_loss_curves(show: bool = True) -> str:
    """Loss curves for each signal combination."""
    data = _load(2)

    fig, ax = plt.subplots(figsize=(9, 5))
    for model_key, records in data.items():
        steps = [r["step"] for r in records]
        losses = [r["loss"] for r in records]
        ax.plot(steps, losses,
                label=LABELS.get(model_key, model_key),
                color=COLORS.get(model_key, None),
                linewidth=2)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Exp 2 – Signal Ablation: Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _savefig(fig, "exp2_loss_curves.png", show)


def plot_exp2_gate_weights(layer: int = 0, show: bool = True) -> str:
    """
    Evolution of learned gate signal weights over training steps.
    Shows w_surprise, w_relevance, w_contiguity for models that have them.
    """
    data = _load(2)

    weight_keys = [f"layer{layer}_w_surprise",
                   f"layer{layer}_w_relevance",
                   f"layer{layer}_w_contiguity"]
    weight_labels = ["w_surprise", "w_relevance", "w_contiguity"]

    models_with_weights = {
        k: v for k, v in data.items()
        if any(wk in v[0] for wk in weight_keys)
    }

    n = len(models_with_weights)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (model_key, records) in zip(axes, models_with_weights.items()):
        steps = [r["step"] for r in records]
        for wk, wl in zip(weight_keys, weight_labels):
            vals = [r.get(wk, 0.0) for r in records]
            ax.plot(steps, vals, label=wl, linewidth=2)
        ax.set_title(LABELS.get(model_key, model_key))
        ax.set_xlabel("Step")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Gate Weight")
    fig.suptitle(f"Exp 2 – Gate Signal Weights Evolution (Layer {layer})", y=1.02)
    fig.tight_layout()
    return _savefig(fig, f"exp2_gate_weights_layer{layer}.png", show)


def plot_exp2_gate_bias(layer: int = 0, show: bool = True) -> str:
    """Gate bias and decay lambda evolution per model."""
    data = _load(2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for model_key, records in data.items():
        steps = [r["step"] for r in records]
        biases = [r.get(f"layer{layer}_gate_bias", None) for r in records]
        decays = [r.get(f"layer{layer}_decay_lambda", None) for r in records]
        label = LABELS.get(model_key, model_key)
        color = COLORS.get(model_key, None)
        if any(b is not None for b in biases):
            ax1.plot(steps, biases, label=label, color=color, linewidth=2)
        if any(d is not None for d in decays):
            ax2.plot(steps, decays, label=label, color=color, linewidth=2)

    ax1.set_title(f"Gate Bias (Layer {layer})")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Gate Bias")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_title(f"Decay Lambda (Layer {layer})")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Decay λ")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Exp 2 – Gate Bias & Decay Lambda Evolution")
    fig.tight_layout()
    return _savefig(fig, f"exp2_gate_bias_decay_layer{layer}.png", show)


# ---------------------------------------------------------------------------
# Experiment 3 – needle-in-haystack retrieval accuracy
# ---------------------------------------------------------------------------

def plot_exp3_accuracy_bar(show: bool = True) -> str:
    """Bar chart of retrieval accuracy for multi_signal vs dual_store."""
    data = _load(3)

    model_keys = list(data.keys())
    accuracies = [data[k]["accuracy"] * 100 for k in model_keys]
    labels = [LABELS.get(k, k) for k in model_keys]
    colors = [COLORS.get(k, "#888") for k in model_keys]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, accuracies, color=colors, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(
        f"Exp 3 – Needle-in-Haystack Retrieval\n"
        f"(seq_len={data[model_keys[0]]['seq_len']}, "
        f"n_eval={data[model_keys[0]]['num_eval']})"
    )
    ax.set_ylim(0, max(accuracies) * 1.3 + 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _savefig(fig, "exp3_accuracy_bar.png", show)


# ---------------------------------------------------------------------------
# Experiment 4 – catastrophic forgetting / continual learning
# ---------------------------------------------------------------------------

def plot_exp4_retention(show: bool = True) -> str:
    """
    Grouped bar chart showing phase1_before, phase1_after, phase2_after loss
    for multi_signal and dual_store, plus retention_ratio annotation.
    """
    data = _load(4)

    phases = ["phase1_before", "phase1_after", "phase2_after"]
    phase_labels = ["Phase 1\n(Before)", "Phase 1\n(After)", "Phase 2\n(After)"]
    model_keys = list(data.keys())

    x = np.arange(len(phases))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, mk in enumerate(model_keys):
        vals = [data[mk][p] for p in phases]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width,
                      label=LABELS.get(mk, mk),
                      color=COLORS.get(mk, None),
                      edgecolor="white")
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels)
    ax.set_ylabel("Loss")
    ax.set_title("Exp 4 – Continual Learning: Phase Losses")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Annotate retention ratios
    for i, mk in enumerate(model_keys):
        ratio = data[mk]["retention_ratio"]
        color = COLORS.get(mk, "#333")
        ax.annotate(
            f"{LABELS.get(mk, mk)}\nretention={ratio:.4f}",
            xy=(0.02 + i * 0.5, 0.97), xycoords="axes fraction",
            va="top", fontsize=8, color=color,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
        )

    fig.tight_layout()
    return _savefig(fig, "exp4_retention.png", show)


def plot_exp4_retention_ratio(show: bool = True) -> str:
    """Bar chart of retention ratios (closer to 1.0 = less forgetting)."""
    data = _load(4)

    model_keys = list(data.keys())
    ratios = [data[k]["retention_ratio"] for k in model_keys]
    labels = [LABELS.get(k, k) for k in model_keys]
    colors = [COLORS.get(k, "#888") for k in model_keys]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, ratios, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.6f", padding=3, fontsize=9)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Ideal (1.0)")
    ax.set_ylabel("Retention Ratio")
    ax.set_title("Exp 4 – Memory Retention Ratio\n(1.0 = no forgetting)")
    ax.legend()
    ax.set_ylim(min(ratios) * 0.9999, max(ratios) * 1.0001)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.5f"))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _savefig(fig, "exp4_retention_ratio.png", show)


# ---------------------------------------------------------------------------
# All-in-one: generate every plot
# ---------------------------------------------------------------------------

def plot_all(show: bool = False) -> list[str]:
    """Generate all plots and save to ./plots/. Returns list of saved paths."""
    paths = [
        plot_exp1_loss_curves(show=show),
        plot_exp1_final_loss_bar(show=show),
        plot_exp2_loss_curves(show=show),
        plot_exp2_gate_weights(layer=0, show=show),
        plot_exp2_gate_bias(layer=0, show=show),
        plot_exp3_accuracy_bar(show=show),
        plot_exp4_retention(show=show),
        plot_exp4_retention_ratio(show=show),
    ]
    print(f"\nAll {len(paths)} plots saved to: {PLOTS_DIR}")
    return paths


if __name__ == "__main__":
    plot_all(show=False)
