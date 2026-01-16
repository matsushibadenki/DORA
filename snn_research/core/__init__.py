# snn_research/core/__init__.py

from .snn_core import SpikingNeuralSubstrate
# 後方互換性が必要な場合はエイリアスを残すことができますが、
# 今回は構造刷新のため削除し、新しいクラスのみを公開します。

# Neuromorphic OS Kernel
from .neuromorphic_os import NeuromorphicOS

__all__ = [
    "SpikingNeuralSubstrate",
    "NeuromorphicOS",
]