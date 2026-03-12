"""
DualStoreMemoryAsContextTransformer - Extending Titans MAC Transformer.

This module provides a transformer architecture that uses the dual-store
memory system in place of the original single NeuralMemory.

The architecture follows the Memory-as-Context (MAC) pattern from Titans:
- Long sequences are segmented into chunks
- Memory provides context for each segment
- Attention operates within segments with optional sliding window

Key differences from original MAC transformer:
- Uses DualStoreMemory (fast + slow) instead of single NeuralMemory
- Multi-signal gating (surprise + goal-relevance + temporal contiguity)
- Periodic consolidation from fast to slow store
"""

from __future__ import annotations
from typing import Callable, Optional, Tuple, Dict
from functools import partial
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, ModuleList, Linear

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from axial_positional_embedding import ContinuousAxialPositionalEmbedding
from rotary_embedding_torch import RotaryEmbedding
from x_transformers.attend import Attend

from .dual_store_memory import DualStoreMemory, SingleStoreMemory, DualStoreMemState
from .multi_signal_memory import MultiSignalMemState
from .config import Config, default_config


# Helper functions
LinearNoBias = partial(Linear, bias=False)


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def divisible_by(num, den):
    return (num % den) == 0


def round_up_multiple(seq, mult):
    from math import ceil
    return ceil(seq / mult) * mult


def pad_at_dim(t, pad, dim=-1, value=0.):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value=value)


def pad_and_segment_with_inverse(
    seq,
    segment_len,
    fold_into_batch=True,
    inverse_remove_pad=True
):
    batch, seq_len = seq.shape[:2]
    next_seq_len_mult = round_up_multiple(seq_len, segment_len)

    padding = next_seq_len_mult - seq_len
    needs_pad = padding > 0

    if needs_pad:
        seq = F.pad(seq, (0, 0, 0, padding))

    if fold_into_batch:
        seq = rearrange(seq, 'b (w n) d -> (b w) n d', n=segment_len)

    def inverse(out):
        if fold_into_batch:
            out = rearrange(out, '(b w) ... n d -> b ... (w n) d', b=batch)
        if needs_pad and inverse_remove_pad:
            out = out[..., :-padding, :]
        return out

    return seq, inverse


class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


def FeedForward(dim, mult=4):
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(
        nn.RMSNorm(dim),
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Linear(dim_inner, dim)
    )


class SegmentedAttention(Module):
    """Segmented attention layer from Titans."""

    def __init__(
        self,
        dim: int,
        segment_len: int,
        dim_head: int = 64,
        heads: int = 8,
        num_persist_mem_tokens: int = 0,
        accept_value_residual: bool = False,
        sliding: bool = False
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim)

        dim_inner = dim_head * heads

        self.rotary_emb = RotaryEmbedding(dim_head)
        self.attend = Attend(causal=True)

        self.to_qkv = LinearNoBias(dim, dim_inner * 3)
        self.to_out = LinearNoBias(dim_inner, dim)

        self.to_learned_v_mix = nn.Sequential(
            nn.Linear(dim, heads),
            Rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        ) if accept_value_residual else None

        self.segment_len = segment_len
        self.sliding = sliding
        self.heads = heads

        self.split_heads = Rearrange('b n (h d) -> b h n d', h=heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.persistent_memory = nn.Parameter(
            torch.zeros(2, heads, num_persist_mem_tokens, dim_head)
        )
        self.num_persist_mem_tokens = num_persist_mem_tokens

    def forward(
        self,
        seq: Tensor,
        value_residual: Optional[Tensor] = None,
        output_gating: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        batch, seq_len = seq.shape[:2]

        seq, inverse_segment = pad_and_segment_with_inverse(
            seq, self.segment_len, fold_into_batch=False
        )

        seq = self.norm(seq)

        q, k, v = self.to_qkv(seq).chunk(3, dim=-1)
        q, k, v = map(self.split_heads, (q, k, v))

        orig_v = v

        if exists(self.to_learned_v_mix) and exists(value_residual):
            mix = self.to_learned_v_mix(seq)
            v = v.lerp(value_residual, mix)

        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        q, k, v = tuple(
            rearrange(t, 'b h (w n) d -> (b w) h n d', n=self.segment_len)
            for t in (q, k, v)
        )

        pmk, pmv = repeat(self.persistent_memory, 'kv ... -> kv b ...', b=k.shape[0])
        k = torch.cat((pmk, k), dim=-2)
        v = torch.cat((pmv, v), dim=-2)

        out, _ = self.attend(q, k, v)

        out = self.merge_heads(out)
        out = self.to_out(out)

        out = rearrange(out, '(b w) n d -> b (w n) d', b=batch)
        out = inverse_segment(out)

        if exists(output_gating):
            out = out * output_gating

        return out, orig_v


class DualStoreMemoryAsContextTransformer(Module):
    """
    Transformer with dual-store memory as context (MAC architecture).

    Extends the original Titans MemoryAsContextTransformer with:
    - Dual-store memory (fast hippocampus + slow neocortex)
    - Multi-signal gating (surprise + goal-relevance + temporal contiguity)
    - Periodic consolidation between stores

    Args:
        num_tokens: Vocabulary size
        dim: Model dimension
        depth: Number of transformer layers
        segment_len: Segment length for attention
        config: Full configuration object
        use_dual_store: If False, uses single-store multi-signal memory
    """

    def __init__(
        self,
        num_tokens: int,
        dim: int,
        depth: int,
        segment_len: int,
        config: Optional[Config] = None,
        dim_head: int = 64,
        heads: int = 4,
        ff_mult: float = 4.0,
        num_persist_mem_tokens: int = 4,
        use_dual_store: bool = True,
        neural_memory_layers: Optional[Tuple[int, ...]] = None
    ):
        super().__init__()

        config = config or default_config

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.axial_pos_emb = ContinuousAxialPositionalEmbedding(dim=dim, num_axial_dims=2)

        self.segment_len = segment_len
        self.use_dual_store = use_dual_store

        # Memory configuration - use memory_heads from config for efficiency
        memory_heads = config.model.memory_heads
        neural_memory_kwargs = dict(
            chunk_size=segment_len,
            heads=memory_heads,
        )

        layers = tuple(range(1, depth + 1))
        neural_memory_layers = default(neural_memory_layers, layers)

        self.layers = ModuleList([])

        for layer in layers:
            is_first = layer == 1

            attn = SegmentedAttention(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                segment_len=segment_len,
                accept_value_residual=not is_first,
                num_persist_mem_tokens=num_persist_mem_tokens
            )

            mem = None
            if layer in neural_memory_layers:
                if use_dual_store:
                    mem = DualStoreMemory(
                        dim=dim,
                        dual_store_config=config.dual_store,
                        gating_config=config.gating,
                        **neural_memory_kwargs
                    )
                else:
                    mem = SingleStoreMemory(
                        dim=dim,
                        gating_config=config.gating,
                        **neural_memory_kwargs
                    )

            ff = FeedForward(dim=dim, mult=ff_mult)

            self.layers.append(ModuleList([mem, attn, ff]))

        self.norm = nn.RMSNorm(dim)
        self.to_logits = LinearNoBias(dim, num_tokens)

        self.register_buffer('zero', torch.tensor(0.), persistent=False)

    def forward(
        self,
        x: Tensor,
        return_loss: bool = False,
        return_metrics: bool = False,
        mem_states: Optional[list] = None
    ) -> Tensor | Tuple[Tensor, Dict]:
        """
        Forward pass through the transformer.

        Args:
            x: Input token indices [batch, seq_len]
            return_loss: If True, compute cross-entropy loss
            return_metrics: If True, return memory metrics
            mem_states: Optional list of memory states for each layer

        Returns:
            logits or loss, optionally with metrics dict
        """
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        batch, seq_len = x.shape

        x = self.token_emb(x)

        pos_emb = self.axial_pos_emb.forward_with_seq_len(
            seq_len, (self.segment_len,)
        )
        x = x + pos_emb

        mem_states = mem_states or [None] * len(self.layers)
        next_mem_states = []

        value_residual = None
        all_metrics = {}

        for layer_idx, (mem, attn, ff) in enumerate(self.layers):
            retrieved = None

            if exists(mem):
                mem_state = mem_states[layer_idx]

                retrieved, next_mem_state, info = mem(
                    x,
                    state=mem_state,
                    return_gating_signals=return_metrics
                )
                next_mem_states.append(next_mem_state)

                if return_metrics and info:
                    layer_metrics = mem.get_metrics()
                    for k, v in layer_metrics.items():
                        all_metrics[f'layer{layer_idx}_{k}'] = v

                x = x + retrieved
            else:
                next_mem_states.append(None)

            attn_out, values = attn(x, value_residual=value_residual)
            value_residual = default(value_residual, values)
            x = x + attn_out

            x = x + ff(x)

        x = self.norm(x)
        logits = self.to_logits(x)

        if not return_loss:
            if return_metrics:
                return logits, all_metrics, next_mem_states
            return logits

        loss = F.cross_entropy(
            rearrange(logits, 'b n l -> b l n'),
            labels
        )

        if return_metrics:
            return loss, all_metrics, next_mem_states
        return loss


class VanillaTransformer(Module):
    """
    Vanilla transformer baseline without memory for comparison.

    Same architecture as DualStoreMemoryAsContextTransformer but without
    any memory modules.
    """

    def __init__(
        self,
        num_tokens: int,
        dim: int,
        depth: int,
        segment_len: int,
        dim_head: int = 64,
        heads: int = 4,
        ff_mult: float = 4.0,
        num_persist_mem_tokens: int = 4
    ):
        super().__init__()

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.axial_pos_emb = ContinuousAxialPositionalEmbedding(dim=dim, num_axial_dims=2)

        self.segment_len = segment_len

        self.layers = ModuleList([])

        for layer in range(1, depth + 1):
            is_first = layer == 1

            attn = SegmentedAttention(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                segment_len=segment_len,
                accept_value_residual=not is_first,
                num_persist_mem_tokens=num_persist_mem_tokens
            )

            ff = FeedForward(dim=dim, mult=ff_mult)

            self.layers.append(ModuleList([attn, ff]))

        self.norm = nn.RMSNorm(dim)
        self.to_logits = LinearNoBias(dim, num_tokens)

    def forward(
        self,
        x: Tensor,
        return_loss: bool = False
    ) -> Tensor:
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        batch, seq_len = x.shape

        x = self.token_emb(x)

        pos_emb = self.axial_pos_emb.forward_with_seq_len(
            seq_len, (self.segment_len,)
        )
        x = x + pos_emb

        value_residual = None

        for attn, ff in self.layers:
            attn_out, values = attn(x, value_residual=value_residual)
            value_residual = default(value_residual, values)
            x = x + attn_out

            x = x + ff(x)

        x = self.norm(x)
        logits = self.to_logits(x)

        if not return_loss:
            return logits

        loss = F.cross_entropy(
            rearrange(logits, 'b n l -> b l n'),
            labels
        )

        return loss


def create_model(
    model_type: str,
    config: Optional[Config] = None,
    **override_kwargs
) -> Module:
    """
    Factory function to create different model variants.

    Args:
        model_type: One of 'vanilla', 'titans_original', 'multi_signal', 'dual_store'
        config: Configuration object
        **override_kwargs: Override specific config values

    Returns:
        Instantiated model
    """
    config = config or default_config

    base_kwargs = dict(
        num_tokens=config.model.num_tokens,
        dim=config.model.dim,
        depth=config.model.depth,
        segment_len=config.model.segment_len,
        dim_head=config.model.dim_head,
        heads=config.model.heads,
        ff_mult=config.model.ff_mult,
    )
    base_kwargs.update(override_kwargs)

    if model_type == 'vanilla':
        return VanillaTransformer(**base_kwargs)

    elif model_type == 'titans_original':
        # Original Titans MAC transformer
        from titans_pytorch import MemoryAsContextTransformer
        return MemoryAsContextTransformer(
            num_tokens=base_kwargs['num_tokens'],
            dim=base_kwargs['dim'],
            depth=base_kwargs['depth'],
            segment_len=base_kwargs['segment_len'],
            dim_head=base_kwargs['dim_head'],
            heads=base_kwargs['heads'],
            ff_mult=base_kwargs['ff_mult'],
        )

    elif model_type == 'multi_signal':
        # Multi-signal gated single store
        return DualStoreMemoryAsContextTransformer(
            config=config,
            use_dual_store=False,
            **base_kwargs
        )

    elif model_type == 'dual_store':
        # Full dual-store architecture
        return DualStoreMemoryAsContextTransformer(
            config=config,
            use_dual_store=True,
            **base_kwargs
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
