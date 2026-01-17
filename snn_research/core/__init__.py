# ファイルパス: snn_research/core/__init__.py
# 日本語タイトル: SNN Core Package Init
# 目的・内容:
#   Neuromorphic OS Kernelと基本的な基盤クラスを公開する。

from .snn_core import SpikingNeuralSubstrate
from .neuromorphic_os import NeuromorphicOS

__all__ = [
    "SpikingNeuralSubstrate",
    "NeuromorphicOS",
]