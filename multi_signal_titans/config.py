"""
Configuration for Multi-Signal Gated Dual-Store Memory Architecture.

This module defines all hyperparameters for the extended Titans architecture,
including parameters for the multi-signal gating function and dual-store memory system.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the transformer model architecture."""

    # Core transformer dimensions
    dim: int = 256
    depth: int = 2
    heads: int = 4
    dim_head: int = 64
    ff_mult: float = 4.0

    # Sequence and segment parameters
    segment_len: int = 128
    num_tokens: int = 256  # For byte-level modeling (enwik8)

    # Memory model parameters
    memory_heads: int = 1  # Memory heads (1 = more efficient, heads = more expressive)
    memory_model_depth: int = 2
    memory_model_expansion: float = 2.0  # Reduced for parameter efficiency


@dataclass
class MultiSignalGatingConfig:
    """
    Configuration for the multi-signal composite gating function.

    The composite gate G_t modulates memory writes using three signals:
    - S_t (surprise): Prediction error from the memory on incoming tokens
    - R_t (goal-relevance): Cosine similarity with learned goal representation
    - C_t (temporal contiguity): Exponentially decaying trace of recent surprises

    Neuroscience motivation:
    - Surprise: Dopaminergic reward prediction error signals
    - Goal-relevance: Prefrontal cortex modulation of hippocampal encoding
    - Temporal contiguity: Synaptic tagging and capture
    """

    # Temporal contiguity window size (k in the formula)
    contiguity_window: int = 8

    # Initial values for learnable gate weights (will be learned via backprop)
    # w_surprise starts small so early surprise values (~4-7 cross-entropy) don't
    # saturate the gate at init. Negative bias keeps gate near 0.5 initially.
    init_w_surprise: float = 0.1
    init_w_relevance: float = 0.1
    init_w_contiguity: float = 0.1
    init_bias: float = -2.0

    # Initial decay rate for temporal contiguity (lambda)
    init_decay_lambda: float = 0.5


@dataclass
class DualStoreConfig:
    """
    Configuration for the dual-store memory architecture.

    Implements Complementary Learning Systems theory:
    - Fast store (M_f): Hippocampus-like, rapid episodic learning
    - Slow store (M_s): Neocortex-like, gradual semantic consolidation

    The consolidation process replays high-attention items from fast to slow store.
    """

    # Consolidation frequency (every T steps)
    consolidation_interval: int = 64

    # Number of top entries to consolidate based on cumulative attention
    consolidation_top_k: int = 16

    # Initial value for the read combination parameter (beta)
    # o_t = beta * M_f_read + (1 - beta) * M_s_read
    init_beta: float = 0.5


@dataclass
class TrainingConfig:
    """Configuration for training procedures."""

    # Basic training parameters
    batch_size: int = 4
    seq_len: int = 1024
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Training duration
    max_steps: int = 10000
    warmup_steps: int = 500

    # Logging and checkpointing
    log_interval: int = 50
    checkpoint_interval: int = 1000
    eval_interval: int = 500

    # Checkpointing paths
    checkpoint_dir: str = "./checkpoints/"  # Primary checkpoint location
    local_checkpoint_dir: str = "./checkpoints/"  # Fallback (same for non-Colab)

    # Mixed precision
    use_amp: bool = True

    # Random seed for reproducibility
    seed: int = 42


@dataclass
class ExperimentConfig:
    """Configuration for specific experiments."""

    # Experiment 1 & 2: enwik8 training
    exp1_steps: int = 10000
    exp2_steps: int = 5000

    # Experiment 3: Needle-in-haystack
    needle_seq_len: int = 4096
    needle_train_examples: int = 500
    needle_eval_examples: int = 100
    needle_pattern_len: int = 8

    # Experiment 4: Continual learning
    continual_phase_steps: int = 5000
    continual_eval_samples: int = 100


@dataclass
class Config:
    """Master configuration combining all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    gating: MultiSignalGatingConfig = field(default_factory=MultiSignalGatingConfig)
    dual_store: DualStoreConfig = field(default_factory=DualStoreConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    # Convenience method to estimate parameter count
    def estimate_params(self) -> int:
        """Rough estimate of model parameters (target: 3-5M)."""
        d = self.model.dim
        depth = self.model.depth
        heads = self.model.heads
        vocab = self.model.num_tokens

        # Embedding
        emb = vocab * d

        # Per layer: attention + feedforward + memory
        attn_per_layer = 4 * d * d  # Q, K, V, O projections
        ff_per_layer = 2 * d * int(d * self.model.ff_mult)
        mem_per_layer = d * int(d * self.model.memory_model_expansion) * 2

        per_layer = attn_per_layer + ff_per_layer + mem_per_layer

        # Total (rough estimate)
        total = emb + depth * per_layer + vocab * d  # + output projection

        return total


# Default configuration instance
default_config = Config()


def get_config(
    dim: Optional[int] = None,
    depth: Optional[int] = None,
    **kwargs
) -> Config:
    """
    Create a configuration with optional overrides.

    Args:
        dim: Override model dimension
        depth: Override model depth
        **kwargs: Additional overrides for nested configs

    Returns:
        Config instance with specified overrides
    """
    config = Config()

    if dim is not None:
        config.model.dim = dim
    if depth is not None:
        config.model.depth = depth

    # Handle nested config updates
    for key, value in kwargs.items():
        if '.' in key:
            parts = key.split('.')
            obj = config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)

    return config
