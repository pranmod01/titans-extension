"""
Tests for the multi-signal gating logic in MultiSignalNeuralMemory.

Covers:
- compute_goal_relevance: output range, shape, gradient flow
- compute_temporal_contiguity: shape, decay behaviour, history update
- compute_composite_gate: output range, signal ablation, no NaN at init
- MultiSignalNeuralMemoryAblation: disabled signals are truly zeroed out
- Forward pass: output shape, no NaN after a gradient step
"""

import pytest
import torch
import torch.nn as nn

from multi_signal_titans.config import MultiSignalGatingConfig
from multi_signal_titans.multi_signal_memory import (
    MultiSignalNeuralMemory,
    MultiSignalNeuralMemoryAblation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DIM = 64
BATCH = 2
SEQ = 32
CHUNK = 16
HEADS = 2


@pytest.fixture
def gating_config():
    return MultiSignalGatingConfig(
        contiguity_window=4,
        init_w_surprise=0.1,
        init_w_relevance=0.1,
        init_w_contiguity=0.1,
        init_bias=-2.0,
        init_decay_lambda=0.5,
    )


@pytest.fixture
def memory(gating_config):
    return MultiSignalNeuralMemory(
        dim=DIM,
        gating_config=gating_config,
        chunk_size=CHUNK,
        heads=HEADS,
    ).eval()


# ---------------------------------------------------------------------------
# compute_goal_relevance
# ---------------------------------------------------------------------------

class TestGoalRelevance:
    def test_output_shape(self, memory):
        h = torch.randn(BATCH, SEQ, DIM)
        r = memory.compute_goal_relevance(h)
        assert r.shape == (BATCH, SEQ)

    def test_output_range(self, memory):
        h = torch.randn(BATCH, SEQ, DIM)
        r = memory.compute_goal_relevance(h)
        assert r.min() >= -1.0 - 1e-5
        assert r.max() <=  1.0 + 1e-5

    def test_no_nan(self, memory):
        h = torch.randn(BATCH, SEQ, DIM)
        r = memory.compute_goal_relevance(h)
        assert not torch.isnan(r).any()

    def test_gradient_flows(self, memory):
        mem = MultiSignalNeuralMemory(
            dim=DIM,
            gating_config=MultiSignalGatingConfig(),
            chunk_size=CHUNK,
            heads=HEADS,
        ).train()
        h = torch.randn(BATCH, SEQ, DIM, requires_grad=True)
        r = mem.compute_goal_relevance(h)
        r.sum().backward()
        assert h.grad is not None
        assert not torch.isnan(h.grad).any()


# ---------------------------------------------------------------------------
# compute_temporal_contiguity
# ---------------------------------------------------------------------------

class TestTemporalContiguity:
    def test_output_shape_no_history(self, memory):
        surprise = torch.rand(BATCH, SEQ)
        c, history = memory.compute_temporal_contiguity(surprise, surprise_history=None)
        assert c.shape == (BATCH, SEQ)
        assert history.shape == (BATCH, memory.contiguity_window)

    def test_output_shape_3d_surprise(self, memory):
        # surprise can be [batch, heads, seq] from NeuralMemory
        surprise = torch.rand(BATCH, HEADS, SEQ)
        c, history = memory.compute_temporal_contiguity(surprise, surprise_history=None)
        assert c.shape == (BATCH, SEQ)

    def test_history_used(self, memory):
        surprise = torch.rand(BATCH, SEQ)
        k = memory.contiguity_window
        history = torch.zeros(BATCH, k)
        c_no_hist, _ = memory.compute_temporal_contiguity(surprise, surprise_history=None)
        c_with_hist, _ = memory.compute_temporal_contiguity(surprise, surprise_history=history)
        # Zero history should give same result as no history
        assert torch.allclose(c_no_hist, c_with_hist, atol=1e-5)

    def test_non_negative(self, memory):
        surprise = torch.rand(BATCH, SEQ).abs()
        c, _ = memory.compute_temporal_contiguity(surprise)
        assert (c >= 0).all()

    def test_no_nan(self, memory):
        surprise = torch.rand(BATCH, SEQ)
        c, history = memory.compute_temporal_contiguity(surprise)
        assert not torch.isnan(c).any()
        assert not torch.isnan(history).any()

    def test_history_detached(self, memory):
        surprise = torch.rand(BATCH, SEQ)
        _, history = memory.compute_temporal_contiguity(surprise)
        assert not history.requires_grad


# ---------------------------------------------------------------------------
# compute_composite_gate
# ---------------------------------------------------------------------------

class TestCompositeGate:
    def _signals(self):
        surprise = torch.rand(BATCH, SEQ) * 6  # realistic early-training range
        relevance = torch.rand(BATCH, SEQ) * 2 - 1  # cosine sim in [-1, 1]
        contiguity = torch.rand(BATCH, SEQ) * 3
        return surprise, relevance, contiguity

    def test_output_shape(self, memory):
        s, r, c = self._signals()
        g = memory.compute_composite_gate(s, r, c)
        assert g.shape == (BATCH, SEQ)

    def test_output_range(self, memory):
        s, r, c = self._signals()
        g = memory.compute_composite_gate(s, r, c)
        assert g.min() >= 0.0 - 1e-6
        assert g.max() <= 1.0 + 1e-6

    def test_no_nan_at_init(self, memory):
        # Specifically test with the high surprise values that used to saturate gate
        s = torch.full((BATCH, SEQ), 6.0)  # typical early cross-entropy
        r = torch.zeros(BATCH, SEQ)
        c = torch.zeros(BATCH, SEQ)
        g = memory.compute_composite_gate(s, r, c)
        assert not torch.isnan(g).any()

    def test_not_saturated_at_init(self, memory):
        # With fixed init (w=0.1, bias=-2), gate should not be stuck at 0 or 1
        s = torch.full((BATCH, SEQ), 5.0)
        r = torch.zeros(BATCH, SEQ)
        c = torch.zeros(BATCH, SEQ)
        g = memory.compute_composite_gate(s, r, c)
        # gate logit = 0.1*5 + 0 + 0 + (-2) = -1.5 -> sigmoid ≈ 0.18
        assert (g > 0.05).all(), "gate is fully closed"
        assert (g < 0.95).all(), "gate is saturated open"

    def test_3d_surprise_averaged(self, memory):
        s3d = torch.rand(BATCH, HEADS, SEQ)
        s2d = s3d.mean(dim=1)
        r = torch.zeros(BATCH, SEQ)
        c = torch.zeros(BATCH, SEQ)
        g3d = memory.compute_composite_gate(s3d, r, c)
        g2d = memory.compute_composite_gate(s2d, r, c)
        assert torch.allclose(g3d, g2d, atol=1e-5)

    def test_higher_surprise_increases_gate(self, memory):
        r = torch.zeros(BATCH, SEQ)
        c = torch.zeros(BATCH, SEQ)
        g_low  = memory.compute_composite_gate(torch.zeros(BATCH, SEQ), r, c)
        g_high = memory.compute_composite_gate(torch.full((BATCH, SEQ), 5.0), r, c)
        # w_surprise > 0, so higher surprise should raise gate values
        assert (g_high >= g_low).all()


# ---------------------------------------------------------------------------
# MultiSignalNeuralMemoryAblation
# ---------------------------------------------------------------------------

class TestAblation:
    @pytest.mark.parametrize("use_s,use_r,use_c", [
        (True,  False, False),
        (False, True,  False),
        (False, False, True),
        (True,  True,  True),
        (False, False, False),
    ])
    def test_disabled_weights_are_zero(self, gating_config, use_s, use_r, use_c):
        mem = MultiSignalNeuralMemoryAblation(
            dim=DIM,
            use_surprise=use_s,
            use_relevance=use_r,
            use_contiguity=use_c,
            gating_config=gating_config,
            chunk_size=CHUNK,
            heads=HEADS,
        )
        if not use_s:
            assert mem.w_surprise.item() == 0.0
            assert not mem.w_surprise.requires_grad
        if not use_r:
            assert mem.w_relevance.item() == 0.0
            assert not mem.w_relevance.requires_grad
        if not use_c:
            assert mem.w_contiguity.item() == 0.0
            assert not mem.w_contiguity.requires_grad

    def test_surprise_only_gate_ignores_relevance(self, gating_config):
        mem = MultiSignalNeuralMemoryAblation(
            dim=DIM,
            use_surprise=True, use_relevance=False, use_contiguity=False,
            gating_config=gating_config, chunk_size=CHUNK, heads=HEADS,
        ).eval()
        s = torch.rand(BATCH, SEQ)
        r_a = torch.rand(BATCH, SEQ)
        r_b = torch.rand(BATCH, SEQ)
        c   = torch.rand(BATCH, SEQ)
        g_a = mem.compute_composite_gate(s, r_a, c)
        g_b = mem.compute_composite_gate(s, r_b, c)
        assert torch.allclose(g_a, g_b), "gate should be identical when relevance is disabled"

    def test_output_in_range(self, gating_config):
        mem = MultiSignalNeuralMemoryAblation(
            dim=DIM,
            use_surprise=True, use_relevance=True, use_contiguity=True,
            gating_config=gating_config, chunk_size=CHUNK, heads=HEADS,
        ).eval()
        s = torch.rand(BATCH, SEQ) * 7
        r = torch.rand(BATCH, SEQ) * 2 - 1
        c = torch.rand(BATCH, SEQ) * 3
        g = mem.compute_composite_gate(s, r, c)
        assert g.min() >= 0.0 - 1e-6
        assert g.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Forward pass (integration)
# ---------------------------------------------------------------------------

class TestForwardPass:
    def test_output_shape(self, memory):
        seq = torch.randn(BATCH, SEQ, DIM)
        retrieved, next_state, gating_signals = memory(seq, return_gating_signals=True)
        assert retrieved.shape == (BATCH, SEQ, DIM)
        assert gating_signals is not None

    def test_no_nan_output(self, memory):
        seq = torch.randn(BATCH, SEQ, DIM)
        retrieved, _, _ = memory(seq)
        assert not torch.isnan(retrieved).any()

    def test_state_carries_over(self, memory):
        seq = torch.randn(BATCH, SEQ, DIM)
        _, state1, _ = memory(seq)
        _, state2, _ = memory(seq, state=state1)
        assert state2 is not None

    def test_no_nan_after_gradient_step(self, gating_config):
        mem = MultiSignalNeuralMemory(
            dim=DIM,
            gating_config=gating_config,
            chunk_size=CHUNK,
            heads=HEADS,
        ).train()
        # Use SGD to avoid AdamW's in-place ops conflicting with titans-pytorch's
        # shared fast-weight parameter storage.
        optimizer = torch.optim.SGD(mem.parameters(), lr=1e-3)

        for _ in range(3):
            seq = torch.randn(BATCH, SEQ, DIM)
            retrieved, _, _ = mem(seq)
            loss = retrieved.pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mem.parameters(), 1.0)
            optimizer.step()

        assert not torch.isnan(retrieved).any()
        assert not torch.isnan(loss).any()
