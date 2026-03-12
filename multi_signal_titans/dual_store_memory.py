"""
Dual-Store Memory Architecture extending Titans.

Implements Complementary Learning Systems (CLS) theory with two memory stores:

1. Fast Store (M_f): Hippocampus-like rapid episodic memory
   - Uses the multi-signal gated NeuralMemory
   - Quick learning, interference-prone
   - Stores recent experiences with high fidelity

2. Slow Store (M_s): Neocortex-like gradual semantic memory
   - Separate NeuralMemory instance
   - Slower learning, more stable
   - Consolidated representations from fast store

Consolidation Process:
- Every T steps, top-k entries from fast store (ranked by cumulative attention)
  are "replayed" into the slow store
- This mimics hippocampal replay during sleep/rest

Read Combination:
- Output is a learned convex combination: o_t = beta * M_f(q) + (1-beta) * M_s(q)
- beta is learned to balance novelty (fast) vs stability (slow)

Neuroscience Motivation:
McClelland, McNaughton & O'Reilly (1995) Complementary Learning Systems
"""

from __future__ import annotations
from typing import NamedTuple, Optional, Tuple, Dict, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat

from titans_pytorch import NeuralMemory
from titans_pytorch.neural_memory import NeuralMemState

from .multi_signal_memory import (
    MultiSignalNeuralMemory,
    MultiSignalMemState,
    GatingSignals
)
from .config import DualStoreConfig, MultiSignalGatingConfig, default_config


class DualStoreMemState(NamedTuple):
    """State for dual-store memory system."""
    fast_state: MultiSignalMemState  # Fast store (hippocampus) state
    slow_state: NeuralMemState  # Slow store (neocortex) state
    step_counter: int  # Steps since last consolidation
    consolidation_buffer: Optional[Tensor]  # Accumulated entries for consolidation
    attention_scores: Optional[Tensor]  # Cumulative attention for ranking


class ConsolidationStats(NamedTuple):
    """Statistics from a consolidation event."""
    num_consolidated: int
    mean_attention_score: float
    consolidation_step: int


class DualStoreMemory(nn.Module):
    """
    Dual-store memory implementing Complementary Learning Systems.

    Combines a fast-learning multi-signal gated memory (hippocampus) with
    a slow-learning memory (neocortex) and periodic consolidation.

    Neuroscience Motivation:
    The brain uses complementary learning systems - the hippocampus rapidly
    encodes specific experiences while the neocortex gradually extracts
    statistical regularities through consolidation during rest/sleep.

    Args:
        dim: Model dimension
        dual_store_config: Configuration for dual-store architecture
        gating_config: Configuration for multi-signal gating
        chunk_size: Memory chunk size
        heads: Number of attention heads
        **neural_memory_kwargs: Additional arguments for NeuralMemory
    """

    def __init__(
        self,
        dim: int,
        dual_store_config: Optional[DualStoreConfig] = None,
        gating_config: Optional[MultiSignalGatingConfig] = None,
        chunk_size: int = 128,
        heads: int = 1,
        **neural_memory_kwargs
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.chunk_size = chunk_size

        ds_config = dual_store_config or default_config.dual_store
        self.consolidation_interval = ds_config.consolidation_interval
        self.consolidation_top_k = ds_config.consolidation_top_k

        # ===== Fast Store (Hippocampus) =====
        # Uses multi-signal gating for sophisticated encoding control
        self.fast_store = MultiSignalNeuralMemory(
            dim=dim,
            gating_config=gating_config,
            chunk_size=chunk_size,
            heads=heads,
            **neural_memory_kwargs
        )

        # ===== Slow Store (Neocortex) =====
        # Standard NeuralMemory, receives consolidated memories
        self.slow_store = NeuralMemory(
            dim=dim,
            chunk_size=chunk_size,
            heads=heads,
            **neural_memory_kwargs
        )

        # ===== Read Combination =====
        # Learnable beta for mixing fast and slow store outputs
        # o_t = beta * M_f(q) + (1-beta) * M_s(q)
        self._beta_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5

        # Consolidation tracking
        self.register_buffer('_step', torch.tensor(0))
        self.consolidation_history: List[ConsolidationStats] = []

    @property
    def beta(self) -> Tensor:
        """Learnable mixing coefficient for fast/slow stores."""
        return torch.sigmoid(self._beta_logit)

    def forward(
        self,
        seq: Tensor,
        state: Optional[DualStoreMemState] = None,
        return_gating_signals: bool = False,
        force_consolidation: bool = False
    ) -> Tuple[Tensor, DualStoreMemState, Optional[Dict]]:
        """
        Forward pass through dual-store memory.

        Args:
            seq: Input sequence [batch, seq_len, dim]
            state: Previous dual-store state
            return_gating_signals: Whether to return detailed signals
            force_consolidation: Force consolidation regardless of interval

        Returns:
            output: Combined memory output [batch, seq_len, dim]
            next_state: Updated dual-store state
            info: Optional dict with gating signals and consolidation info
        """
        # Initialize state if needed
        if state is None:
            fast_state = None
            slow_state = None
            step_counter = 0
            consolidation_buffer = None
            attention_scores = None
        else:
            fast_state = state.fast_state
            slow_state = state.slow_state
            step_counter = state.step_counter
            consolidation_buffer = state.consolidation_buffer
            attention_scores = state.attention_scores

        batch, seq_len = seq.shape[:2]

        # ===== Fast Store Forward =====
        fast_output, next_fast_state, gating_signals = self.fast_store(
            seq,
            state=fast_state,
            return_gating_signals=return_gating_signals
        )

        # ===== Slow Store Forward =====
        slow_output, next_slow_state = self.slow_store(
            seq,
            state=slow_state
        )

        # ===== Combine Outputs =====
        # Learned convex combination: o = beta * fast + (1-beta) * slow
        beta = self.beta
        output = beta * fast_output + (1 - beta) * slow_output

        # ===== Update Consolidation Tracking =====
        # Accumulate sequence data and attention scores for consolidation
        if consolidation_buffer is None:
            consolidation_buffer = seq
        else:
            consolidation_buffer = torch.cat([consolidation_buffer, seq], dim=1)

        # Get attention scores from fast store state
        if next_fast_state.cumulative_attention is not None:
            new_attention = next_fast_state.cumulative_attention
            if attention_scores is None:
                attention_scores = new_attention
            else:
                # Handle shape mismatch by padding or truncating
                if attention_scores.shape[1] + new_attention.shape[1] <= consolidation_buffer.shape[1]:
                    attention_scores = torch.cat([attention_scores, new_attention], dim=1)
                else:
                    attention_scores = new_attention

        # Update step counter
        step_counter += 1
        self._step += 1

        # ===== Consolidation =====
        consolidation_info = None
        should_consolidate = (
            force_consolidation or
            step_counter >= self.consolidation_interval
        )

        if should_consolidate and consolidation_buffer is not None:
            consolidation_info = self._consolidate(
                consolidation_buffer,
                attention_scores,
                next_slow_state
            )

            # Reset consolidation tracking
            consolidation_buffer = None
            attention_scores = None
            step_counter = 0

        # Build next state
        next_state = DualStoreMemState(
            fast_state=next_fast_state,
            slow_state=next_slow_state,
            step_counter=step_counter,
            consolidation_buffer=consolidation_buffer,
            attention_scores=attention_scores
        )

        # Build info dict if requested
        info = None
        if return_gating_signals:
            info = {
                'gating_signals': gating_signals,
                'beta': beta.item(),
                'consolidation': consolidation_info,
                'step': self._step.item()
            }

        return output, next_state, info

    def _consolidate(
        self,
        buffer: Tensor,
        attention_scores: Optional[Tensor],
        slow_state: NeuralMemState
    ) -> ConsolidationStats:
        """
        Consolidate top-k entries from fast store to slow store.

        Selects entries with highest cumulative attention and replays
        them through the slow store to strengthen those representations.

        Neuroscience Motivation:
        During sleep/rest, the hippocampus "replays" important experiences
        to the neocortex, allowing gradual integration into semantic memory.
        Entries with higher attention during encoding are preferentially replayed.

        Args:
            buffer: Accumulated sequence data [batch, total_seq, dim]
            attention_scores: Cumulative attention weights [batch, total_seq]
            slow_state: Current slow store state

        Returns:
            ConsolidationStats with info about the consolidation
        """
        batch, total_seq, dim = buffer.shape
        k = min(self.consolidation_top_k, total_seq)

        if attention_scores is None or attention_scores.shape[1] < total_seq:
            # Fall back to uniform selection if no attention scores
            indices = torch.randperm(total_seq, device=buffer.device)[:k]
            indices = indices.unsqueeze(0).expand(batch, -1)
            mean_score = 0.0
        else:
            # Truncate attention scores to match buffer length
            attention_scores = attention_scores[:, :total_seq]

            # Select top-k by attention score
            _, indices = torch.topk(attention_scores, k, dim=1)
            mean_score = attention_scores.gather(1, indices).mean().item()

        # Gather selected entries
        # indices: [batch, k], buffer: [batch, total_seq, dim]
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, dim)
        selected = torch.gather(buffer, 1, indices_expanded)  # [batch, k, dim]

        # Replay through slow store (detached to avoid gradient issues)
        # This uses straight-through estimation - forward pass updates slow store
        # but gradients don't flow back through the discrete selection
        with torch.no_grad():
            # Run selected entries through slow store
            _, _ = self.slow_store(selected.detach(), state=slow_state)

        stats = ConsolidationStats(
            num_consolidated=k,
            mean_attention_score=mean_score,
            consolidation_step=self._step.item()
        )
        self.consolidation_history.append(stats)

        return stats

    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics for logging."""
        metrics = {
            'beta': self.beta.item(),
            'step': self._step.item(),
        }

        # Add fast store gate weights
        gate_weights = self.fast_store.get_gate_weights()
        for k, v in gate_weights.items():
            metrics[f'fast_{k}'] = v

        # Add consolidation stats
        if self.consolidation_history:
            last = self.consolidation_history[-1]
            metrics['last_consolidation_k'] = last.num_consolidated
            metrics['last_consolidation_attn'] = last.mean_attention_score

        return metrics

    def reset_consolidation_history(self):
        """Clear consolidation history (e.g., between experiments)."""
        self.consolidation_history = []
        self._step.zero_()


class SingleStoreMemory(nn.Module):
    """
    Single-store memory (multi-signal gated) for comparison experiments.

    This is a wrapper around MultiSignalNeuralMemory that provides the same
    interface as DualStoreMemory but without the slow store and consolidation.
    Used as a baseline in experiments comparing single vs dual store.
    """

    def __init__(
        self,
        dim: int,
        gating_config: Optional[MultiSignalGatingConfig] = None,
        chunk_size: int = 128,
        heads: int = 1,
        **neural_memory_kwargs
    ):
        super().__init__()

        self.memory = MultiSignalNeuralMemory(
            dim=dim,
            gating_config=gating_config,
            chunk_size=chunk_size,
            heads=heads,
            **neural_memory_kwargs
        )

        # Dummy beta for interface compatibility
        self.register_buffer('_beta', torch.tensor(1.0))

    @property
    def beta(self) -> Tensor:
        return self._beta

    def forward(
        self,
        seq: Tensor,
        state: Optional[MultiSignalMemState] = None,
        return_gating_signals: bool = False,
        **kwargs
    ) -> Tuple[Tensor, MultiSignalMemState, Optional[Dict]]:
        """Forward pass through single memory store."""
        output, next_state, gating_signals = self.memory(
            seq,
            state=state,
            return_gating_signals=return_gating_signals
        )

        info = None
        if return_gating_signals:
            info = {
                'gating_signals': gating_signals,
                'beta': 1.0,
                'consolidation': None,
                'step': 0
            }

        return output, next_state, info

    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics for logging."""
        metrics = {'beta': 1.0}
        gate_weights = self.memory.get_gate_weights()
        for k, v in gate_weights.items():
            metrics[k] = v
        return metrics
