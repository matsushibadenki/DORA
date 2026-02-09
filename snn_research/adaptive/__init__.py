# directory: snn_research/adaptive
# file: __init__.py
# purpose: Adaptive module exports
# description: 適応型学習モジュールの公開インターフェース

from .active_inference_agent import ActiveInferenceAgent
from .intrinsic_motivator import IntrinsicMotivator
from .on_chip_self_corrector import OnChipSelfCorrector
from .test_time_adaptation import TimeAdaptationWrapper

# Backward compatibility alias (if needed temporarily)
TestTimeAdaptationWrapper = TimeAdaptationWrapper 

__all__ = [
    "ActiveInferenceAgent",
    "IntrinsicMotivator",
    "OnChipSelfCorrector",
    "TimeAdaptationWrapper",
    "TestTimeAdaptationWrapper" # Export alias
]