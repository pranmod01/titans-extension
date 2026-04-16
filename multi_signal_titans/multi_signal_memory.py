"""
Multi-Signal Gated Neural Memory extending Titans NeuralMemory.

This module implements a composite gating function that modulates memory writes
using three biologically-inspired signals:

1. Surprise (S_t): The original Titans surprise metric - memory's prediction loss
   on incoming tokens. Motivated by dopaminergic reward prediction error signals.

2. Goal-Relevance (R_t): Cosine similarity between current hidden state and a
   learned goal representation. Motivated by prefrontal cortex modulation of
   hippocampal encoding.

3. Temporal Contiguity (C_t): Exponentially decaying trace of recent surprises.
   Motivated by synaptic tagging and capture - events near surprising moments
   get tagged for consolidation.

The composite gate is: G_t = sigmoid(w_s * S_t + w_r * R_t + w_c * C_t + b)
"""

from __future__ import annotations
from typing import NamedTuple, Optional, Tuple, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat

from titans_pytorch import NeuralMemory
from titans_pytorch.neural_memory import NeuralMemState, mem_state_detach

from .config import MultiSignalGatingConfig, default_config


@dataclass
class GatingSignals:
    """Container for the three gating signals and composite gate."""
    surprise: Tensor  # S_t: prediction error
    relevance: Tensor  # R_t: goal similarity
    contiguity: Tensor  # C_t: temporal trace
    composite_gate: Tensor  # G_t: final gate value


class MultiSignalMemState(NamedTuple):
    """Extended memory state including multi-signal gating state."""
    neural_mem_state: NeuralMemState
    surprise_history: Tensor  # Rolling buffer of recent surprises for C_t
    cumulative_attention: Tensor  # Running sum of attention weights for consolidation


class MultiSignalNeuralMemory(nn.Module):
    """
    Neural memory with multi-signal composite gating.

    Wraps the original Titans NeuralMemory and replaces the surprise-only gating
    with a composite function combining surprise, goal-relevance, and temporal
    contiguity signals.

    Neuroscience Motivation:
    - The combination of these signals mirrors how the brain gates memory encoding:
      surprising events (dopamine), goal-relevant information (PFC), and temporal
      context (hippocampal time cells) all contribute to what gets remembered.

    Args:
        dim: Model dimension
        gating_config: Configuration for the multi-signal gating
        **neural_memory_kwargs: Additional arguments passed to NeuralMemory
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

        self.dim = dim
        self.heads = heads
        self.chunk_size = chunk_size
        config = gating_config or default_config.gating

        # Underlying NeuralMemory from titans-pytorch
        self.neural_memory = NeuralMemory(
            dim=dim,
            chunk_size=chunk_size,
            heads=heads,
            **neural_memory_kwargs
        )

        # ===== Multi-Signal Gating Components =====

        # Learnable gate weights for composite function
        # G_t = sigmoid(w_s * S_t + w_r * R_t + w_c * C_t + b)
        self.w_surprise = nn.Parameter(torch.tensor(config.init_w_surprise))
        self.w_relevance = nn.Parameter(torch.tensor(config.init_w_relevance))
        self.w_contiguity = nn.Parameter(torch.tensor(config.init_w_contiguity))
        self.gate_bias = nn.Parameter(torch.tensor(config.init_bias))

        # Goal representation vector for R_t (goal-relevance signal)
        # Neuroscience: Represents prefrontal cortex goal state
        self.goal_vector = nn.Parameter(torch.randn(dim) * 0.02)

        # Learnable decay rate for temporal contiguity
        # C_t = max over j in [t-k, t-1] of S_j * exp(-lambda * (t-j))
        self.decay_lambda = nn.Parameter(torch.tensor(config.init_decay_lambda))

        # Contiguity window size (k in the formula)
        self.contiguity_window = config.contiguity_window

        # Projection to normalize hidden states for goal similarity
        self.state_proj = nn.Linear(dim, dim, bias=False)

    @torch.amp.autocast('cuda', enabled=False)
    def compute_goal_relevance(self, hidden_states: Tensor) -> Tensor:
        """
        Compute goal-relevance signal R_t.

        R_t = cosine_similarity(h_t, g_t) where g_t is the learned goal vector.

        Neuroscience Motivation:
        The prefrontal cortex maintains goal representations that modulate
        hippocampal encoding - information relevant to current goals is
        preferentially encoded into memory.

        Args:
            hidden_states: Current hidden states [batch, seq, dim]

        Returns:
            Goal relevance scores [batch, seq]
        """
        # Project and normalize hidden states
        h_proj = self.state_proj(hidden_states)
        h_norm = F.normalize(h_proj, dim=-1, eps=1e-8)

        # Normalize goal vector
        g_norm = F.normalize(self.goal_vector, dim=-1, eps=1e-8)

        # Cosine similarity
        relevance = torch.einsum('...d,d->...', h_norm, g_norm)

        return relevance

    @torch.amp.autocast('cuda', enabled=False)
    def compute_temporal_contiguity(
        self,
        current_surprise: Tensor,
        surprise_history: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute temporal contiguity signal C_t.

        C_t = max over j in [t-k, t-1] of S_j * exp(-lambda * (t-j))

        This creates an exponentially decaying trace of recent surprises,
        so events occurring near surprising moments get enhanced gating.

        Neuroscience Motivation:
        Synaptic tagging and capture - synapses active near the time of
        a surprising/important event get "tagged" and are more likely to
        be consolidated into long-term memory.

        Args:
            current_surprise: Current surprise values [batch, seq] or [batch, heads, seq]
            surprise_history: Previous surprise buffer [batch, k] or None

        Returns:
            Contiguity signal [batch, seq], updated surprise history [batch, k]
        """
        # Handle different input shapes - always work in float32 to avoid NaN from AMP
        if current_surprise.dim() == 3:
            surprise = current_surprise.float().mean(dim=1)
        else:
            surprise = current_surprise.float()

        surprise = torch.clamp(surprise, min=0.0, max=100.0)

        batch, seq_len = surprise.shape
        k = self.contiguity_window
        device = surprise.device

        # Initialize history if needed
        if surprise_history is None:
            surprise_history = torch.zeros(batch, k, device=device, dtype=torch.float32)
        else:
            surprise_history = surprise_history.float()

        # Compute decay weights: exp(-lambda * j) for j = 1, 2, ..., k
        decay_lambda = F.softplus(self.decay_lambda.float())  # Ensure positive
        j = torch.arange(1, k + 1, device=device, dtype=torch.float32)
        decay_weights = torch.exp(-decay_lambda * j)  # [k]

        # Vectorised rolling-max via a causal convolution view.
        # Build a buffer [batch, seq_len + k] of all surprises (history then current).
        # Then use unfold to get windows of size k for each output position.
        full = torch.cat([surprise_history, surprise], dim=1)  # [batch, seq_len + k]

        # full[:, t : t+k] are the k *preceding* surprises for output position t
        # unfold gives [batch, seq_len, k]
        windows = full.unfold(dimension=1, size=k, step=1)[:, :seq_len, :]  # [batch, seq_len, k]

        # decay_weights[j] corresponds to lag j+1; flip so index 0 = oldest lag k
        # windows[:, t, 0] = surprise at t-k, windows[:, t, k-1] = surprise at t-1
        dw = decay_weights.flip(0)  # [k], index 0 = lag k, index k-1 = lag 1

        weighted = windows * dw.unsqueeze(0).unsqueeze(0)  # [batch, seq_len, k]
        contiguity = weighted.max(dim=-1).values  # [batch, seq_len]

        # Update history: keep the last k surprises
        new_history = full[:, -k:].detach()

        return contiguity, new_history

    @torch.amp.autocast('cuda', enabled=False)
    def compute_composite_gate(
        self,
        surprise: Tensor,
        relevance: Tensor,
        contiguity: Tensor
    ) -> Tensor:
        """
        Compute the composite gating signal G_t.

        G_t = sigmoid(w_s * S_t + w_r * R_t + w_c * C_t + b)

        Args:
            surprise: Surprise signal [batch, seq] or [batch, heads, seq]
            relevance: Goal-relevance signal [batch, seq]
            contiguity: Temporal contiguity signal [batch, seq]

        Returns:
            Composite gate values [batch, seq] in (0, 1)
        """
        # Average surprise over heads if needed — always use float32 to stay
        # numerically stable inside AMP autocast regions.
        if surprise.dim() == 3:
            s = surprise.float().mean(dim=1)  # [batch, seq]
        else:
            s = surprise.float()

        # Clamp surprise to prevent inf/NaN from AMP or early training instability
        s = torch.clamp(s, min=0.0, max=100.0)

        # Normalize signals to similar scales
        # Surprise is already a loss value, relevance is cosine sim [-1, 1]
        # Shift relevance to [0, 1] range
        r = (relevance.float() + 1) / 2

        # Contiguity is already non-negative; clamp for safety
        c = torch.clamp(contiguity.float(), min=0.0, max=100.0)

        # Composite gate
        gate_logits = (
            self.w_surprise * s +
            self.w_relevance * r +
            self.w_contiguity * c +
            self.gate_bias
        )

        return torch.sigmoid(gate_logits)

    @torch.amp.autocast('cuda', enabled=False)
    def forward(
        self,
        seq: Tensor,
        state: Optional[MultiSignalMemState] = None,
        return_gating_signals: bool = False,
        **kwargs
    ) -> Tuple[Tensor, MultiSignalMemState, Optional[GatingSignals]]:
        """
        Forward pass with multi-signal gated memory.

        Args:
            seq: Input sequence [batch, seq_len, dim] or with views
            state: Previous memory state
            return_gating_signals: Whether to return detailed gating info
            **kwargs: Additional arguments for NeuralMemory

        Returns:
            retrieved: Retrieved memory content [batch, seq_len, dim]
            next_state: Updated memory state
            gating_signals: Optional GatingSignals if return_gating_signals=True
        """
        # Handle state
        if state is not None:
            neural_mem_state = state.neural_mem_state
            surprise_history = state.surprise_history
            cumulative_attention = state.cumulative_attention
        else:
            neural_mem_state = None
            surprise_history = None
            cumulative_attention = None

        # Get sequence shape
        if seq.dim() == 4:
            # Different views for qkv: [views, batch, seq, dim]
            batch, seq_len = seq.shape[1:3]
            hidden_for_relevance = seq[0]  # Use first view
        else:
            batch, seq_len = seq.shape[:2]
            hidden_for_relevance = seq

        # Forward through underlying NeuralMemory with surprise tracking
        retrieved, next_neural_mem_state, surprises = self.neural_memory(
            seq,
            state=neural_mem_state,
            return_surprises=True,
            **kwargs
        )

        # Extract surprise from returned values
        # surprises is (unweighted_mem_model_loss, adaptive_lr)
        unweighted_loss, adaptive_lr = surprises
        # surprise shape depends on chunking: [batch, heads, chunk_seq]
        surprise = unweighted_loss

        # Get the actual surprise sequence length (may differ due to chunking)
        surprise_seq_len = surprise.shape[-1] if surprise is not None else seq_len

        # Compute goal-relevance signal for the effective sequence
        # Truncate or pad to match surprise length
        if hidden_for_relevance.shape[1] > surprise_seq_len:
            hidden_truncated = hidden_for_relevance[:, :surprise_seq_len]
        else:
            hidden_truncated = hidden_for_relevance
        relevance = self.compute_goal_relevance(hidden_truncated)

        # Ensure relevance matches surprise_seq_len
        if relevance.shape[1] < surprise_seq_len:
            # Pad with zeros
            pad_len = surprise_seq_len - relevance.shape[1]
            relevance = F.pad(relevance, (0, pad_len), value=0.0)
        elif relevance.shape[1] > surprise_seq_len:
            relevance = relevance[:, :surprise_seq_len]

        # Compute temporal contiguity
        contiguity, new_surprise_history = self.compute_temporal_contiguity(
            surprise, surprise_history
        )

        # Ensure contiguity matches relevance length
        if contiguity.shape[1] < relevance.shape[1]:
            pad_len = relevance.shape[1] - contiguity.shape[1]
            contiguity = F.pad(contiguity, (0, pad_len), value=0.0)
        elif contiguity.shape[1] > relevance.shape[1]:
            contiguity = contiguity[:, :relevance.shape[1]]

        # Compute composite gate
        composite_gate = self.compute_composite_gate(surprise, relevance, contiguity)

        # Expand gate to match retrieved sequence length
        if composite_gate.shape[1] < seq_len:
            # Interpolate or repeat to match output size
            composite_gate = F.interpolate(
                composite_gate.unsqueeze(1),
                size=seq_len,
                mode='nearest'
            ).squeeze(1)
        elif composite_gate.shape[1] > seq_len:
            composite_gate = composite_gate[:, :seq_len]

        # Apply gate to modulate retrieved memories
        # Cast gate back to retrieved's dtype so the residual stream stays in float16 under AMP
        gate_expanded = composite_gate.to(retrieved.dtype).unsqueeze(-1)  # [batch, seq, 1]
        retrieved = retrieved * gate_expanded

        # Track cumulative attention for consolidation
        # Using the surprise-weighted adaptive_lr as proxy for attention
        if cumulative_attention is None:
            cumulative_attention = adaptive_lr.mean(dim=1)  # [batch, seq]
        else:
            # Handle shape mismatch
            new_attention = adaptive_lr.mean(dim=1)
            # Just use the new attention for simplicity
            cumulative_attention = new_attention

        # Build next state
        next_state = MultiSignalMemState(
            neural_mem_state=next_neural_mem_state,
            surprise_history=new_surprise_history,
            cumulative_attention=cumulative_attention
        )

        if return_gating_signals:
            # Return signals truncated to min common length for logging
            min_len = min(surprise.shape[-1] if surprise is not None else 0,
                          relevance.shape[-1], contiguity.shape[-1])
            gating_signals = GatingSignals(
                surprise=surprise.mean(dim=1)[:, :min_len] if surprise is not None else None,
                relevance=relevance[:, :min_len],
                contiguity=contiguity[:, :min_len],
                composite_gate=composite_gate[:, :min_len]
            )
            return retrieved, next_state, gating_signals

        return retrieved, next_state, None

    def get_gate_weights(self) -> Dict[str, float]:
        """Return current learned gate weights for logging."""
        return {
            'w_surprise': self.w_surprise.item(),
            'w_relevance': self.w_relevance.item(),
            'w_contiguity': self.w_contiguity.item(),
            'gate_bias': self.gate_bias.item(),
            'decay_lambda': F.softplus(self.decay_lambda).item()
        }


class MultiSignalNeuralMemoryAblation(MultiSignalNeuralMemory):
    """
    Ablation variant of MultiSignalNeuralMemory for experiments.

    Allows selectively disabling individual gating signals to measure
    their contribution.

    Args:
        use_surprise: Whether to include surprise signal
        use_relevance: Whether to include goal-relevance signal
        use_contiguity: Whether to include temporal contiguity signal
    """

    def __init__(
        self,
        dim: int,
        use_surprise: bool = True,
        use_relevance: bool = True,
        use_contiguity: bool = True,
        **kwargs
    ):
        super().__init__(dim, **kwargs)

        self.use_surprise = use_surprise
        self.use_relevance = use_relevance
        self.use_contiguity = use_contiguity

        # Zero out weights for disabled signals
        if not use_surprise:
            self.w_surprise = nn.Parameter(torch.zeros(1).squeeze().clone(), requires_grad=False)
        if not use_relevance:
            self.w_relevance = nn.Parameter(torch.zeros(1).squeeze().clone(), requires_grad=False)
        if not use_contiguity:
            self.w_contiguity = nn.Parameter(torch.zeros(1).squeeze().clone(), requires_grad=False)

    @torch.amp.autocast('cuda', enabled=False)
    def compute_composite_gate(
        self,
        surprise: Tensor,
        relevance: Tensor,
        contiguity: Tensor
    ) -> Tensor:
        """Compute gate with ablated signals zeroed."""
        if surprise.dim() == 3:
            s = surprise.float().mean(dim=1)
        else:
            s = surprise.float()

        s = torch.clamp(s, min=0.0, max=100.0)

        r = (relevance.float() + 1) / 2
        c = torch.clamp(contiguity.float(), min=0.0, max=100.0)

        # Apply ablation masks
        if not self.use_surprise:
            s = torch.zeros_like(s)
        if not self.use_relevance:
            r = torch.zeros_like(r)
        if not self.use_contiguity:
            c = torch.zeros_like(c)

        gate_logits = (
            self.w_surprise * s +
            self.w_relevance * r +
            self.w_contiguity * c +
            self.gate_bias
        )

        return torch.sigmoid(gate_logits)
