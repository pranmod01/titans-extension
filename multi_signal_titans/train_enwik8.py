"""
Training script for Multi-Signal Gated Dual-Store Memory Architecture on enwik8.

Features:
- Automatic checkpointing to Google Drive (for Colab resilience)
- Auto-resume from checkpoints
- Mixed precision training (AMP)
- Comprehensive metric logging to JSON
- tqdm progress bars
- Best model saving based on validation loss
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

from .config import Config, TrainingConfig, default_config
from .transformer import create_model, count_parameters


# ============================================================================
# Environment Detection
# ============================================================================

def is_colab() -> bool:
    """Check if running in Google Colab."""
    return os.path.exists('/content')


def setup_drive():
    """Mount Google Drive if in Colab."""
    if is_colab():
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("Google Drive mounted successfully.")
            return True
        except Exception as e:
            print(f"Failed to mount Google Drive: {e}")
            return False
    return False


def get_checkpoint_dir(config: TrainingConfig) -> Path:
    """Get appropriate checkpoint directory based on environment."""
    if is_colab() and os.path.exists('/content/drive/MyDrive'):
        path = Path(config.checkpoint_dir)
    else:
        path = Path(config.local_checkpoint_dir)

    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================================
# Dataset
# ============================================================================

class Enwik8Dataset(Dataset):
    """enwik8 byte-level language modeling dataset."""

    def __init__(
        self,
        data: torch.Tensor,
        seq_len: int,
        split: str = 'train'
    ):
        self.data = data
        self.seq_len = seq_len
        self.split = split

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        return self.data[idx:idx + self.seq_len + 1].long()


def load_enwik8(
    data_dir: str = './data',
    seq_len: int = 1024
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load and prepare enwik8 dataset.

    Downloads if not present, splits into train/val/test.
    """
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    enwik8_path = data_path / 'enwik8'

    if not enwik8_path.exists():
        print("Downloading enwik8...")
        import urllib.request
        import zipfile

        url = 'http://mattmahoney.net/dc/enwik8.zip'
        zip_path = data_path / 'enwik8.zip'

        urllib.request.urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)

        zip_path.unlink()
        print("Download complete.")

    with open(enwik8_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)

    data = torch.from_numpy(data.copy())

    # Standard splits: 90M train, 5M val, 5M test
    n = len(data)
    train_data = data[:int(0.9 * n)]
    val_data = data[int(0.9 * n):int(0.95 * n)]
    test_data = data[int(0.95 * n):]

    return train_data, val_data, test_data


def create_dataloaders(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: TrainingConfig
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""

    train_dataset = Enwik8Dataset(train_data, config.seq_len, 'train')
    val_dataset = Enwik8Dataset(val_data, config.seq_len, 'val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader


# ============================================================================
# Checkpointing
# ============================================================================

class CheckpointManager:
    """Manages checkpointing and resumption."""

    def __init__(
        self,
        checkpoint_dir: Path,
        experiment_name: str = 'default'
    ):
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        self.metrics_file = checkpoint_dir / f'{experiment_name}_metrics.json'
        self.checkpoint_file = checkpoint_dir / f'{experiment_name}_checkpoint.pt'
        self.best_model_file = checkpoint_dir / f'{experiment_name}_best.pt'

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[GradScaler],
        step: int,
        epoch: int,
        running_loss: float,
        metrics_history: list,
        rng_state: dict,
        best_val_loss: float
    ):
        """Save a complete checkpoint."""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'step': step,
            'epoch': epoch,
            'running_loss': running_loss,
            'best_val_loss': best_val_loss,
            'rng_state': rng_state,
            'timestamp': datetime.now().isoformat()
        }

        torch.save(checkpoint, self.checkpoint_file)

        # Save metrics separately as JSON
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics_history, f, indent=2)

        print(f"Checkpoint saved at step {step}")

    def save_best_model(self, model: nn.Module, val_loss: float, step: int):
        """Save the best model based on validation loss."""
        torch.save({
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'step': step,
            'timestamp': datetime.now().isoformat()
        }, self.best_model_file)
        print(f"New best model saved (val_loss: {val_loss:.4f})")

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[GradScaler],
        device: torch.device
    ) -> Tuple[int, int, float, list, float]:
        """
        Load checkpoint if exists.

        Returns:
            step, epoch, running_loss, metrics_history, best_val_loss
        """
        if not self.checkpoint_file.exists():
            return 0, 0, 0.0, [], float('inf')

        print(f"Loading checkpoint from {self.checkpoint_file}")
        checkpoint = torch.load(self.checkpoint_file, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scaler and checkpoint['scaler_state_dict']:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Restore RNG state
        rng_state = checkpoint['rng_state']
        torch.set_rng_state(rng_state['torch'])
        np.random.set_state(rng_state['numpy'])
        random.setstate(rng_state['python'])
        if 'cuda' in rng_state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state['cuda'])

        # Load metrics history
        metrics_history = []
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                metrics_history = json.load(f)

        step = checkpoint['step']
        epoch = checkpoint['epoch']
        running_loss = checkpoint['running_loss']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"Resuming from step {step}")

        return step, epoch, running_loss, metrics_history, best_val_loss

    @staticmethod
    def get_rng_state() -> dict:
        """Capture current RNG state."""
        state = {
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
            'python': random.getstate(),
        }
        if torch.cuda.is_available():
            state['cuda'] = torch.cuda.get_rng_state_all()
        return state


# ============================================================================
# Training Loop
# ============================================================================

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    checkpoint_manager: CheckpointManager,
    device: torch.device,
    start_step: int = 0,
    start_epoch: int = 0,
    running_loss: float = 0.0,
    metrics_history: list = None,
    best_val_loss: float = float('inf')
) -> list:
    """
    Main training loop with checkpointing and metrics logging.

    Returns:
        metrics_history: List of all logged metrics
    """
    metrics_history = metrics_history or []

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    scaler = GradScaler() if config.use_amp and device.type == 'cuda' else None

    # Reload optimizer state if resuming
    if start_step > 0:
        _, _, _, _, _ = checkpoint_manager.load_checkpoint(
            model, optimizer, scaler, device
        )

    model.train()

    step = start_step
    epoch = start_epoch
    accumulated_loss = running_loss
    num_accumulated = 0

    # Calculate total batches
    total_steps = config.max_steps

    pbar = tqdm(total=total_steps, initial=start_step, desc="Training")

    while step < total_steps:
        for batch in train_loader:
            if step >= total_steps:
                break

            batch = batch.to(device)

            optimizer.zero_grad()

            with autocast(enabled=config.use_amp and device.type == 'cuda'):
                if hasattr(model, 'forward') and 'return_metrics' in model.forward.__code__.co_varnames:
                    loss, metrics, _ = model(batch, return_loss=True, return_metrics=True)
                else:
                    loss = model(batch, return_loss=True)
                    metrics = {}

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

            accumulated_loss += loss.item()
            num_accumulated += 1
            step += 1

            # Logging
            if step % config.log_interval == 0:
                avg_loss = accumulated_loss / num_accumulated

                log_entry = {
                    'step': step,
                    'loss': avg_loss,
                    'epoch': epoch,
                    'timestamp': datetime.now().isoformat()
                }

                # Add model-specific metrics
                if metrics:
                    log_entry.update(metrics)

                metrics_history.append(log_entry)

                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'epoch': epoch
                })

                accumulated_loss = 0.0
                num_accumulated = 0

            # Validation
            if step % config.eval_interval == 0:
                val_loss = evaluate(model, val_loader, device, config.use_amp)

                log_entry = {
                    'step': step,
                    'val_loss': val_loss,
                    'type': 'validation',
                    'timestamp': datetime.now().isoformat()
                }
                metrics_history.append(log_entry)

                print(f"\nStep {step}: val_loss = {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_manager.save_best_model(model, val_loss, step)

                model.train()

            # Checkpointing
            if step % config.checkpoint_interval == 0:
                checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step,
                    epoch=epoch,
                    running_loss=accumulated_loss,
                    metrics_history=metrics_history,
                    rng_state=CheckpointManager.get_rng_state(),
                    best_val_loss=best_val_loss
                )

            pbar.update(1)

        epoch += 1

    pbar.close()

    # Final checkpoint
    checkpoint_manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        step=step,
        epoch=epoch,
        running_loss=accumulated_loss,
        metrics_history=metrics_history,
        rng_state=CheckpointManager.get_rng_state(),
        best_val_loss=best_val_loss
    )

    return metrics_history


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool = True
) -> float:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            batch = batch.to(device)

            with autocast(enabled=use_amp and device.type == 'cuda'):
                if hasattr(model, 'forward') and 'return_metrics' in model.forward.__code__.co_varnames:
                    loss, _, _ = model(batch, return_loss=True, return_metrics=False)
                else:
                    loss = model(batch, return_loss=True)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


# ============================================================================
# Main Entry Point
# ============================================================================

def main(args=None):
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='Train Multi-Signal Titans')
    parser.add_argument('--model', type=str, default='dual_store',
                       choices=['vanilla', 'titans_original', 'multi_signal', 'dual_store'])
    parser.add_argument('--experiment', type=str, default='default')
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--seq_len', type=int, default=None)
    parser.add_argument('--dim', type=int, default=None)
    parser.add_argument('--depth', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--data_dir', type=str, default='./data')

    args = parser.parse_args(args)

    # Setup
    if is_colab():
        setup_drive()

    config = default_config

    # Apply overrides
    if args.max_steps:
        config.training.max_steps = args.max_steps
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.seq_len:
        config.training.seq_len = args.seq_len
    if args.dim:
        config.model.dim = args.dim
    if args.depth:
        config.model.depth = args.depth
    if args.no_amp:
        config.training.use_amp = False

    config.training.seed = args.seed

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    train_data, val_data, _ = load_enwik8(args.data_dir, config.training.seq_len)
    train_loader, val_loader = create_dataloaders(train_data, val_data, config.training)

    # Create model
    model = create_model(args.model, config)
    model = model.to(device)

    num_params = count_parameters(model)
    print(f"Model: {args.model}")
    print(f"Parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # Checkpoint manager
    checkpoint_dir = get_checkpoint_dir(config.training)
    checkpoint_manager = CheckpointManager(checkpoint_dir, args.experiment)

    # Check for existing checkpoint
    scaler = GradScaler() if config.training.use_amp and device.type == 'cuda' else None
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )

    start_step, start_epoch, running_loss, metrics_history, best_val_loss = \
        checkpoint_manager.load_checkpoint(model, optimizer, scaler, device)

    # Train
    metrics_history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.training,
        checkpoint_manager=checkpoint_manager,
        device=device,
        start_step=start_step,
        start_epoch=start_epoch,
        running_loss=running_loss,
        metrics_history=metrics_history,
        best_val_loss=best_val_loss
    )

    print("Training complete!")
    print(f"Metrics saved to: {checkpoint_manager.metrics_file}")
    print(f"Best model saved to: {checkpoint_manager.best_model_file}")

    return metrics_history


if __name__ == '__main__':
    main()
