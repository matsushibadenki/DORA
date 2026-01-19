# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: ArtificialBrain Shim
# 目的・内容:
#   async_brain_kernel.py に移動した ArtificialBrain への互換性シム。
#   これにより、古いパスを参照しているスクリプトの修正を最小限に抑える。

from snn_research.cognitive_architecture.async_brain_kernel import ArtificialBrain

__all__ = ["ArtificialBrain"]
