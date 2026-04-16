"""
Experiment runners for Multi-Signal Gated Dual-Store Memory Architecture.

Four experiments to validate the architectural contributions:

1. Does multi-signal gating help?
   - Compare: vanilla transformer, original Titans MAC, multi-signal single-store
   - Metric: Training loss curves on enwik8

2. Ablation on gating signals
   - Variants: surprise-only, surprise+relevance, surprise+contiguity, full
   - Metric: Training loss + learned gate weights over time

3. Does dual-store help for long-range retrieval?
   - Task: Needle-in-a-haystack synthetic task
   - Compare: single-store vs dual-store
   - Metric: Retrieval accuracy

4. Continual learning probe
   - Train on enwik8 first half, then second half
   - Evaluate retention on first half after each phase
   - Compare: single-store vs dual-store
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

from .config import Config, default_config
from .transformer import (
    create_model, count_parameters,
    DualStoreMemoryAsContextTransformer, VanillaTransformer
)
from .multi_signal_memory import MultiSignalNeuralMemoryAblation
from .train_enwik8 import (
    load_enwik8, create_dataloaders, is_colab, setup_drive,
    get_checkpoint_dir, CheckpointManager, evaluate
)


# ============================================================================
# Experiment 1: Does Multi-Signal Gating Help?
# ============================================================================

def experiment_1_gating_comparison(
    config: Optional[Config] = None,
    max_steps: int = 10000,
    device: Optional[torch.device] = None
) -> Dict[str, List[Dict]]:
    """
    Compare training loss curves on enwik8 for three models:
    (a) Vanilla transformer baseline (no memory)
    (b) Original Titans MAC (surprise-only gating)
    (c) Multi-signal gated single-store

    Args:
        config: Configuration (uses default if None)
        max_steps: Training steps (default 10k)
        device: Torch device

    Returns:
        Dictionary mapping model name to list of metrics
    """
    print("=" * 60)
    print("Experiment 1: Does Multi-Signal Gating Help?")
    print("=" * 60)

    config = config or default_config
    config.training.max_steps = max_steps
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    train_data, val_data, _ = load_enwik8('./data', config.training.seq_len)
    train_loader, val_loader = create_dataloaders(train_data, val_data, config.training)

    results = {}

    model_types = [
        ('vanilla', 'Vanilla Transformer'),
        ('titans_original', 'Titans MAC (Original)'),
        ('multi_signal', 'Multi-Signal Gated'),
    ]

    for model_type, model_name in model_types:
        print(f"\nTraining: {model_name}")
        print("-" * 40)

        # Set seed for reproducibility
        torch.manual_seed(config.training.seed)
        np.random.seed(config.training.seed)
        random.seed(config.training.seed)

        # Create model
        model = create_model(model_type, config)
        model = model.to(device)

        num_params = count_parameters(model)
        print(f"Parameters: {num_params:,}")

        # Training
        metrics = train_experiment(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config.training,
            device=device,
            model_name=model_name
        )

        results[model_type] = metrics

        # Clean up
        del model
        torch.cuda.empty_cache()

    # Save results
    save_experiment_results(results, 'experiment_1')

    return results


def experiment_2_gating_ablation(
    config: Optional[Config] = None,
    max_steps: int = 5000,
    device: Optional[torch.device] = None
) -> Dict[str, List[Dict]]:
    """
    Ablation study on gating signals.

    Four variants:
    (a) Surprise only (w_r=w_c=0, recovers baseline)
    (b) Surprise + goal-relevance (w_c=0)
    (c) Surprise + temporal contiguity (w_r=0)
    (d) Full composite (all signals)

    Logs learned gate weights over training to see signal importance.

    Args:
        config: Configuration
        max_steps: Training steps (default 5k)
        device: Torch device

    Returns:
        Dictionary mapping variant name to metrics
    """
    print("=" * 60)
    print("Experiment 2: Gating Signal Ablation")
    print("=" * 60)

    config = config or default_config
    config.training.max_steps = max_steps
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    train_data, val_data, _ = load_enwik8('./data', config.training.seq_len)
    train_loader, val_loader = create_dataloaders(train_data, val_data, config.training)

    results = {}

    # Define ablation variants
    variants = [
        ('surprise_only', True, False, False),
        ('surprise_relevance', True, True, False),
        ('surprise_contiguity', True, False, True),
        ('full_composite', True, True, True),
    ]

    for variant_name, use_s, use_r, use_c in variants:
        print(f"\nTraining: {variant_name}")
        print("-" * 40)

        torch.manual_seed(config.training.seed)
        np.random.seed(config.training.seed)
        random.seed(config.training.seed)

        # Create model with ablated signals
        model = create_ablation_model(
            config=config,
            use_surprise=use_s,
            use_relevance=use_r,
            use_contiguity=use_c
        )
        model = model.to(device)

        # Training with gate weight logging
        metrics = train_experiment(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config.training,
            device=device,
            model_name=variant_name,
            log_gate_weights=True
        )

        results[variant_name] = metrics

        del model
        torch.cuda.empty_cache()

    save_experiment_results(results, 'experiment_2')

    return results


def experiment_3_needle_in_haystack(
    config: Optional[Config] = None,
    seq_len: int = 4096,
    num_train: int = 500,
    num_eval: int = 100,
    device: Optional[torch.device] = None
) -> Dict[str, Dict]:
    """
    Needle-in-a-haystack synthetic task for long-range retrieval.

    Generate sequences where a specific token pattern is embedded at a
    random early position and must be retrieved later.

    Compare single-store vs dual-store retrieval accuracy.

    Args:
        config: Configuration
        seq_len: Sequence length (default 4096)
        num_train: Training examples
        num_eval: Evaluation examples
        device: Torch device

    Returns:
        Dictionary with retrieval accuracy for each model type
    """
    print("=" * 60)
    print("Experiment 3: Needle-in-a-Haystack Long-Range Retrieval")
    print("=" * 60)

    config = config or default_config
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate synthetic data
    train_data, train_labels, train_positions = generate_needle_data(
        num_examples=num_train,
        seq_len=seq_len,
        pattern_len=config.experiment.needle_pattern_len
    )

    eval_data, eval_labels, eval_positions = generate_needle_data(
        num_examples=num_eval,
        seq_len=seq_len,
        pattern_len=config.experiment.needle_pattern_len
    )

    results = {}

    for model_type in ['multi_signal', 'dual_store']:
        print(f"\nTraining: {model_type}")
        print("-" * 40)

        torch.manual_seed(config.training.seed)
        np.random.seed(config.training.seed)
        random.seed(config.training.seed)

        model = create_model(
            model_type,
            config,
            segment_len=256  # Smaller segments for longer sequences
        )
        model = model.to(device)

        # Train on needle task
        train_needle_model(
            model=model,
            train_data=train_data.to(device),
            train_labels=train_labels.to(device),
            config=config.training,
            device=device
        )

        # Evaluate retrieval accuracy
        accuracy = evaluate_needle_retrieval(
            model=model,
            eval_data=eval_data.to(device),
            eval_labels=eval_labels.to(device),
            eval_positions=eval_positions,
            device=device
        )

        results[model_type] = {
            'accuracy': accuracy,
            'num_eval': num_eval,
            'seq_len': seq_len
        }

        print(f"Retrieval Accuracy: {accuracy:.2%}")

        del model
        torch.cuda.empty_cache()

    save_experiment_results(results, 'experiment_3')

    return results


def experiment_4_continual_learning(
    config: Optional[Config] = None,
    phase_steps: int = 5000,
    device: Optional[torch.device] = None
) -> Dict[str, Dict]:
    """
    Continual learning probe.

    Train on first half of enwik8, then switch to second half.
    Evaluate perplexity on held-out sample from first half after each phase.

    Hypothesis: Dual-store retains more from phase 1 due to consolidation.

    Args:
        config: Configuration
        phase_steps: Steps per training phase
        device: Torch device

    Returns:
        Dictionary with perplexity results for each model
    """
    print("=" * 60)
    print("Experiment 4: Continual Learning Probe")
    print("=" * 60)

    config = config or default_config
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load full enwik8
    train_data, val_data, _ = load_enwik8('./data', config.training.seq_len)

    # Split into two halves
    mid = len(train_data) // 2
    phase1_data = train_data[:mid]
    phase2_data = train_data[mid:]

    # Held-out evaluation set from phase 1
    eval_size = config.experiment.continual_eval_samples * config.training.seq_len
    phase1_eval = phase1_data[-eval_size:]
    phase1_train = phase1_data[:-eval_size]

    results = {}

    for model_type in ['multi_signal', 'dual_store']:
        print(f"\nTraining: {model_type}")
        print("-" * 40)

        torch.manual_seed(config.training.seed)
        np.random.seed(config.training.seed)
        random.seed(config.training.seed)

        model = create_model(model_type, config)
        model = model.to(device)

        model_results = {
            'phase1_before': None,
            'phase1_after': None,
            'phase2_after': None,
            'retention_ratio': None,
        }

        # Phase 1: Train on first half
        print("Phase 1: Training on first half...")
        phase1_loader, phase1_eval_loader = create_dataloaders(
            phase1_train, phase1_eval, config.training
        )

        # Baseline: evaluate before any training
        phase1_ppl_before = compute_perplexity(model, phase1_eval_loader, device)
        model_results['phase1_before'] = phase1_ppl_before
        print(f"Phase 1 perplexity before training: {phase1_ppl_before:.2f}")

        train_phase(
            model=model,
            train_loader=phase1_loader,
            config=config.training,
            device=device,
            max_steps=phase_steps
        )

        # Evaluate retention on phase 1 data
        phase1_ppl = compute_perplexity(model, phase1_eval_loader, device)
        model_results['phase1_after'] = phase1_ppl
        print(f"Phase 1 perplexity after phase 1: {phase1_ppl:.2f}")

        # Phase 2: Train on second half
        print("Phase 2: Training on second half...")
        phase2_loader, _ = create_dataloaders(
            phase2_data, val_data, config.training
        )

        train_phase(
            model=model,
            train_loader=phase2_loader,
            config=config.training,
            device=device,
            max_steps=phase_steps
        )

        # Re-evaluate on phase 1 data
        phase1_ppl_after = compute_perplexity(model, phase1_eval_loader, device)
        model_results['phase2_after'] = phase1_ppl_after
        print(f"Phase 1 perplexity after phase 2: {phase1_ppl_after:.2f}")

        # Retention ratio: how much of the phase-1 learning survives phase-2 training.
        # = 1 - (degradation / gain), where degradation = phase2_after - phase1_after
        # and gain = phase1_before - phase1_after (how much phase 1 improved things).
        # Simplified: phase1_after / phase2_after  (<=1 = forgetting, =1 = perfect retention)
        gain = phase1_ppl_before - phase1_ppl          # how much phase 1 improved PPL
        degradation = phase1_ppl_after - phase1_ppl    # how much phase 2 hurt phase-1 PPL
        if gain > 0:
            retention = 1.0 - (degradation / gain)    # 1=perfect, 0=all gains lost, <0=worse than init
        else:
            retention = float('nan')                   # model didn't learn in phase 1
        model_results['retention_ratio'] = retention
        print(f"Retention ratio: {retention:.4f}  (1=perfect, 0=all gains lost)")

        results[model_type] = model_results

        del model
        torch.cuda.empty_cache()

    save_experiment_results(results, 'experiment_4')

    return results


# ============================================================================
# Helper Functions
# ============================================================================

def train_experiment(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config,
    device: torch.device,
    model_name: str,
    log_gate_weights: bool = False
) -> List[Dict]:
    """Train model and return metrics."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    warmup_steps = getattr(config, 'warmup_steps', 500)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = None  # AMP disabled — NeuralMemory uses torch.func.grad+vmap which breaks under float16

    model.train()
    metrics = []
    step = 0

    pbar = tqdm(total=config.max_steps, desc=model_name)

    accumulated_loss = 0.0
    num_accumulated = 0

    while step < config.max_steps:
        for batch in train_loader:
            if step >= config.max_steps:
                break

            batch = batch.to(device)
            optimizer.zero_grad()

            with autocast(enabled=False):
                try:
                    loss, model_metrics, _ = model(
                        batch, return_loss=True, return_metrics=True
                    )
                except TypeError:
                    loss = model(batch, return_loss=True)
                    model_metrics = {}

            loss = loss.float()  # ensure float32 regardless of model internals (titans_pytorch uses mixed precision)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

            scheduler.step()

            loss_val = loss.item()
            if loss_val != loss_val:  # NaN check
                print(f"\n[{model_name}] NaN loss at step {step + 1}, aborting.")
                pbar.close()
                return metrics

            accumulated_loss += loss_val
            num_accumulated += 1
            step += 1

            if step % config.log_interval == 0:
                avg_loss = accumulated_loss / num_accumulated

                entry = {
                    'step': step,
                    'loss': avg_loss,
                    'model': model_name
                }

                if log_gate_weights and model_metrics:
                    entry.update(model_metrics)

                metrics.append(entry)

                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                accumulated_loss = 0.0
                num_accumulated = 0

            pbar.update(1)

    pbar.close()
    return metrics


def create_ablation_model(
    config: Config,
    use_surprise: bool,
    use_relevance: bool,
    use_contiguity: bool
) -> nn.Module:
    """Create model with ablated gating signals."""
    from .dual_store_memory import SingleStoreMemory
    from .multi_signal_memory import MultiSignalNeuralMemoryAblation

    # Create base model structure
    model = DualStoreMemoryAsContextTransformer(
        num_tokens=config.model.num_tokens,
        dim=config.model.dim,
        depth=config.model.depth,
        segment_len=config.model.segment_len,
        config=config,
        use_dual_store=False
    )

    # Replace memory modules with ablation variants
    for layer_modules in model.layers:
        mem = layer_modules[0]
        if mem is not None:
            # Access the underlying memory and replace
            ablation_mem = MultiSignalNeuralMemoryAblation(
                dim=config.model.dim,
                use_surprise=use_surprise,
                use_relevance=use_relevance,
                use_contiguity=use_contiguity,
                chunk_size=config.model.segment_len,
                heads=config.model.memory_heads
            )
            mem.memory = ablation_mem

    return model


def generate_needle_data(
    num_examples: int,
    seq_len: int,
    pattern_len: int = 8
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Generate needle-in-haystack synthetic data.

    Each sequence contains random bytes as "haystack", with a
    unique pattern ("needle") inserted at a random early position.
    The task is to predict the pattern at the end of the sequence.
    """
    data = torch.randint(0, 256, (num_examples, seq_len))
    labels = torch.zeros(num_examples, pattern_len, dtype=torch.long)
    positions = []

    for i in range(num_examples):
        # Create unique pattern
        pattern = torch.randint(0, 256, (pattern_len,))
        labels[i] = pattern

        # Insert at random position in first quarter
        pos = random.randint(10, seq_len // 4)
        positions.append(pos)

        data[i, pos:pos + pattern_len] = pattern

        # Add retrieval cue near end
        cue_pos = seq_len - pattern_len - 10
        data[i, cue_pos:cue_pos + 2] = pattern[:2]  # First 2 bytes as cue

    return data, labels, positions


def train_needle_model(
    model: nn.Module,
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    config,
    device: torch.device,
    epochs: int = 10
):
    """Train model on needle-in-haystack task.

    The input sequences already contain the retrieval cue near the end
    (last pattern_len+10 positions). We supervise the model to predict the
    needle tokens at the final pattern_len positions using next-token
    prediction loss restricted to those positions.
    """
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    num_steps = epochs * (len(train_data) // config.batch_size)
    warmup_steps = min(getattr(config, 'warmup_steps', 500), num_steps // 4)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    step = 0

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        for i in range(0, len(train_data), config.batch_size):
            batch = train_data[i:i + config.batch_size]
            target = train_labels[i:i + config.batch_size]

            optimizer.zero_grad()

            # Use next-token prediction: input is seq[:-1], target is seq[1:]
            # but we only compute loss on the final pattern_len positions so
            # the model is trained specifically to recall the needle at the end.
            inp = batch[:, :-1]
            pattern_len = target.shape[1]

            try:
                logits = model(inp, return_metrics=False)
            except TypeError:
                logits = model(inp)

            # logits: [batch, seq-1, vocab]
            # supervise only the last pattern_len positions
            pred_logits = logits[:, -pattern_len:, :]  # [batch, pattern_len, vocab]

            loss = F.cross_entropy(
                pred_logits.reshape(-1, pred_logits.shape[-1]),
                target.reshape(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            step += 1

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


def evaluate_needle_retrieval(
    model: nn.Module,
    eval_data: torch.Tensor,
    eval_labels: torch.Tensor,
    eval_positions: List[int],
    device: torch.device
) -> float:
    """Evaluate needle retrieval accuracy (per-token).

    Returns the fraction of individual pattern tokens predicted correctly
    across all evaluation examples. Per-token accuracy is more informative
    than exact-sequence match for an 8-token pattern over a 256-token vocab
    (chance is 1/256 ≈ 0.4% per token vs ~0% for exact match).
    """
    model.eval()
    correct_tokens = 0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, len(eval_data), 4):  # small batches to save memory
            batch = eval_data[i:i+4]
            target = eval_labels[i:i+4]

            inp = batch[:, :-1]
            pattern_len = target.shape[1]

            try:
                logits = model(inp, return_metrics=False)
            except TypeError:
                logits = model(inp)

            pred = logits[:, -pattern_len:, :].argmax(dim=-1)  # [batch, pattern_len]

            correct_tokens += (pred.cpu() == target.cpu()).sum().item()
            total_tokens += target.numel()

    return correct_tokens / total_tokens


def train_phase(
    model: nn.Module,
    train_loader: DataLoader,
    config,
    device: torch.device,
    max_steps: int
):
    """Train for one phase of continual learning."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    warmup_steps = getattr(config, 'warmup_steps', 500)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    step = 0

    pbar = tqdm(total=max_steps, desc="Training phase")

    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break

            batch = batch.to(device)
            optimizer.zero_grad()

            try:
                loss, _, _ = model(batch, return_loss=True, return_metrics=False)
            except TypeError:
                loss = model(batch, return_loss=True)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            step += 1
            pbar.update(1)

    pbar.close()


def compute_perplexity(
    model: nn.Module,
    eval_loader: DataLoader,
    device: torch.device
) -> float:
    """Compute perplexity on evaluation set."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in eval_loader:
            batch = batch.to(device)

            try:
                loss, _, _ = model(batch, return_loss=True, return_metrics=False)
            except TypeError:
                loss = model(batch, return_loss=True)

            total_loss += loss.item() * batch.shape[0] * (batch.shape[1] - 1)
            total_tokens += batch.shape[0] * (batch.shape[1] - 1)

    avg_loss = total_loss / total_tokens
    return np.exp(avg_loss)


def save_experiment_results(results: Dict, experiment_name: str):
    """Save experiment results to JSON."""
    if is_colab():
        base_dir = Path('/content/drive/MyDrive/titans_experiments')
    else:
        base_dir = Path('./experiment_results')

    base_dir.mkdir(parents=True, exist_ok=True)

    filename = base_dir / f'{experiment_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

    # Convert tensors to lists for JSON serialization
    def convert(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(filename, 'w') as f:
        json.dump(convert(results), f, indent=2)

    print(f"Results saved to: {filename}")


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Main entry point for running experiments."""
    parser = argparse.ArgumentParser(description='Run Multi-Signal Titans Experiments')
    parser.add_argument('--experiment', type=int, required=True,
                       choices=[1, 2, 3, 4], help='Experiment number to run')
    parser.add_argument('--max_steps', type=int, default=None,
                       help='Override max training steps')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Setup
    if is_colab():
        setup_drive()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = default_config
    config.training.seed = args.seed

    if args.experiment == 1:
        max_steps = args.max_steps or config.experiment.exp1_steps
        experiment_1_gating_comparison(config, max_steps, device)

    elif args.experiment == 2:
        max_steps = args.max_steps or config.experiment.exp2_steps
        experiment_2_gating_ablation(config, max_steps, device)

    elif args.experiment == 3:
        experiment_3_needle_in_haystack(config, device=device)

    elif args.experiment == 4:
        phase_steps = args.max_steps or config.experiment.continual_phase_steps
        experiment_4_continual_learning(config, phase_steps, device)


if __name__ == '__main__':
    main()
