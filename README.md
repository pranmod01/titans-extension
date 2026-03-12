# Multi-Signal Gated Dual-Store Memory Architecture

A research prototype extending the [titans-pytorch](https://github.com/lucidrains/titans-pytorch) library with biologically-inspired memory mechanisms.

## Overview

This project implements two key modifications to the Titans architecture:

### 1. Multi-Signal Gating Function

The original Titans NeuralMemory gates memory writes using a single surprise signal. We replace this with a composite gating function:

```
G_t = σ(w_s · S_t + w_r · R_t + w_c · C_t + b)
```

Where:
- **S_t (Surprise)**: The original Titans surprise metric — prediction error of the memory on incoming tokens
  - *Neuroscience motivation*: Dopaminergic reward prediction error signals

- **R_t (Goal-Relevance)**: Cosine similarity between current hidden state and a learned goal representation
  - *Neuroscience motivation*: Prefrontal cortex modulation of hippocampal encoding

- **C_t (Temporal Contiguity)**: Exponentially decaying trace of recent surprises
  - *Neuroscience motivation*: Synaptic tagging and capture

### 2. Dual-Store Memory Architecture

Implements Complementary Learning Systems (CLS) theory:

- **Fast Store (M_f)**: Hippocampus-like rapid episodic memory using multi-signal gated NeuralMemory
- **Slow Store (M_s)**: Neocortex-like gradual semantic memory with separate parameters
- **Consolidation**: Every T steps, top-k entries from M_f (ranked by cumulative attention) are replayed into M_s
- **Read Combination**: `o_t = β · M_f(q_t) + (1-β) · M_s(q_t)` where β is learned

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
multi_signal_titans/
├── __init__.py           # Package exports
├── config.py             # Hyperparameters and configuration
├── multi_signal_memory.py # MultiSignalNeuralMemory class
├── dual_store_memory.py   # DualStoreMemory class
├── transformer.py         # DualStoreMemoryAsContextTransformer
├── train_enwik8.py        # Training script with checkpointing
└── experiments.py         # All 4 experiment runners
```

## Quick Start

### Local Development

```python
from multi_signal_titans import create_model, Config

# Create dual-store model
config = Config()
model = create_model('dual_store', config)

# Or create single-store with multi-signal gating
model = create_model('multi_signal', config)

# Or vanilla transformer baseline
model = create_model('vanilla', config)
```

### Google Colab

```python
# Cell 1: Mount drive and setup
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/YOUR_USERNAME/titans-extension.git /content/titans-extension
%cd /content/titans-extension
!pip install -q -r requirements.txt

# Cell 2: Run Experiment 1 (gating comparison)
!python -m multi_signal_titans.experiments --experiment 1

# Cell 3: Run Experiment 2 (ablation study)
!python -m multi_signal_titans.experiments --experiment 2

# Cell 4: Run Experiment 3 (needle-in-haystack)
!python -m multi_signal_titans.experiments --experiment 3

# Cell 5: Run Experiment 4 (continual learning)
!python -m multi_signal_titans.experiments --experiment 4
```

## Experiments

### Experiment 1: Does Multi-Signal Gating Help?

Compares training loss curves on enwik8:
- (a) Vanilla transformer (no memory)
- (b) Original Titans MAC (surprise-only gating)
- (c) Multi-signal gated single-store

```bash
python -m multi_signal_titans.experiments --experiment 1 --max_steps 10000
```

### Experiment 2: Gating Signal Ablation

Four variants of the gating function:
- (a) Surprise only
- (b) Surprise + goal-relevance
- (c) Surprise + temporal contiguity
- (d) Full composite

Logs learned gate weights over training.

```bash
python -m multi_signal_titans.experiments --experiment 2 --max_steps 5000
```

### Experiment 3: Long-Range Retrieval

Needle-in-a-haystack synthetic task (4096 token sequences). Compares single-store vs dual-store retrieval accuracy.

```bash
python -m multi_signal_titans.experiments --experiment 3
```

### Experiment 4: Continual Learning Probe

Train on enwik8 first half, then second half. Evaluate retention on first half after each phase.

**Hypothesis**: Dual-store retains more from phase 1 due to consolidation.

```bash
python -m multi_signal_titans.experiments --experiment 4 --max_steps 5000
```

## Training Features

- **Auto-resume**: Automatically resumes from checkpoints on Colab disconnections
- **Google Drive checkpointing**: Saves to `/content/drive/MyDrive/titans_checkpoints/`
- **Mixed precision**: Uses torch.cuda.amp for memory efficiency
- **Best model saving**: Tracks validation loss and saves best checkpoint
- **Metric logging**: All metrics saved to JSON for later plotting

## Model Configuration

Default configuration (target ~3-5M parameters, fits in T4 VRAM):

```python
ModelConfig:
    dim = 256
    depth = 2
    heads = 4
    segment_len = 128

TrainingConfig:
    batch_size = 4
    seq_len = 1024
    learning_rate = 3e-4
    use_amp = True  # Mixed precision
```

## Architecture Details

### MultiSignalNeuralMemory

```python
class MultiSignalNeuralMemory(nn.Module):
    """
    Wraps NeuralMemory with composite gating:
    - goal_vector: Learned PFC-like goal representation
    - w_surprise, w_relevance, w_contiguity: Learned weights
    - decay_lambda: Learned decay rate for temporal trace
    """
```

### DualStoreMemory

```python
class DualStoreMemory(nn.Module):
    """
    Implements CLS theory:
    - fast_store: MultiSignalNeuralMemory (hippocampus)
    - slow_store: Standard NeuralMemory (neocortex)
    - consolidation: Top-k replay every T steps
    - beta: Learned read combination weight
    """
```

## Neuroscience Background

This architecture draws from several neuroscience theories:

1. **Dopaminergic Reward Prediction Error**: Surprise signals in basal ganglia modulate learning rates
2. **Prefrontal Goal Modulation**: PFC maintains goal representations that bias hippocampal encoding
3. **Synaptic Tagging and Capture**: Events near important moments get "tagged" for consolidation
4. **Complementary Learning Systems**: Fast hippocampal learning + slow neocortical integration

## References

- Titans: Learning to Memorize at Test Time (Google Research)
- McClelland, McNaughton & O'Reilly (1995) - Complementary Learning Systems
- Frey & Morris (1997) - Synaptic Tagging and Capture
- Miller & Cohen (2001) - Prefrontal Cortex and Cognitive Control

## License

MIT
