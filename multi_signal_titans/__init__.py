"""
Multi-Signal Gated Dual-Store Memory Architecture extending Titans.

A research prototype implementing biologically-inspired memory mechanisms:
- Multi-signal gating (surprise + goal-relevance + temporal contiguity)
- Dual-store memory (fast hippocampus + slow neocortex)
- Periodic consolidation between stores

Based on:
- Titans: Learning to Memorize at Test Time (Google Research)
- Complementary Learning Systems (McClelland, McNaughton & O'Reilly)
"""

from .config import Config, default_config, get_config
from .multi_signal_memory import (
    MultiSignalNeuralMemory,
    MultiSignalNeuralMemoryAblation,
    MultiSignalMemState,
    GatingSignals
)
from .dual_store_memory import (
    DualStoreMemory,
    SingleStoreMemory,
    DualStoreMemState,
    ConsolidationStats
)
from .transformer import (
    DualStoreMemoryAsContextTransformer,
    VanillaTransformer,
    create_model,
    count_parameters
)

__version__ = "0.1.0"
__author__ = "Research Prototype"

__all__ = [
    # Config
    'Config',
    'default_config',
    'get_config',
    # Memory
    'MultiSignalNeuralMemory',
    'MultiSignalNeuralMemoryAblation',
    'MultiSignalMemState',
    'GatingSignals',
    'DualStoreMemory',
    'SingleStoreMemory',
    'DualStoreMemState',
    'ConsolidationStats',
    # Transformer
    'DualStoreMemoryAsContextTransformer',
    'VanillaTransformer',
    'create_model',
    'count_parameters',
]
