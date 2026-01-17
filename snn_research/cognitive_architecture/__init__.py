# ファイルパス: snn_research/cognitive_architecture/__init__.py
# 日本語タイトル: Cognitive Architecture Init (Cleaned)
# 目的・内容:
#   Neuromorphic OSに必要なコンポーネントのみをエクスポートする。
#   古い巨大なアーキテクチャ（HybridPerceptionCortex等）への依存を断ち切り、
#   ImportErrorを防ぐ。

# --- Core Components for Neuromorphic OS ---
from .global_workspace import GlobalWorkspace
from .astrocyte_network import AstrocyteNetwork
from .neuromorphic_scheduler import NeuromorphicScheduler

# --- Optional / Legacy Components (Needed only if explicitly imported) ---
# エラーの原因となる深い依存関係を持つモジュールは、ここでは自動インポートしない
# from .hybrid_perception_cortex import HybridPerceptionCortex  # DISABLED to prevent ImportError
# from .perception_cortex import PerceptionCortex
# from .motor_cortex import MotorCortex

# 必要に応じて有効化するコンポーネント
from .hippocampus import Hippocampus
# from .prefrontal_cortex import PrefrontalCortex
# from .basal_ganglia import BasalGanglia
# from .amygdala import Amygdala

__all__ = [
    "GlobalWorkspace",
    "AstrocyteNetwork",
    "NeuromorphicScheduler",
    "Hippocampus",
]