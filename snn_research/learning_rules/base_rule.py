# ファイルパス: snn_research/learning_rules/base_rule.py
# 日本語タイトル: Base Plasticity Rule Interface (Fixed)
# 目的・内容:
#   学習則の基底クラス。
#   既存のコードが 'BioLearningRule' を参照しても動くようにエイリアスを追加。

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import torch

class PlasticityRule(ABC):
    """
    Abstract base class for synaptic plasticity rules.
    全ての学習則（STDP, Forward-Forward, Active Inferenceなど）はこのクラスを継承する。
    """

    @abstractmethod
    def update(
        self, 
        pre_spikes: torch.Tensor, 
        post_spikes: torch.Tensor, 
        current_weights: torch.Tensor, 
        local_state: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """
        Calculate weight updates based on local activity.

        Args:
            pre_spikes (Tensor): Spikes from pre-synaptic neurons (Batch, N_pre)
            post_spikes (Tensor): Spikes from post-synaptic neurons (Batch, N_post)
            current_weights (Tensor): Current synaptic weights (N_post, N_pre)
            local_state (Dict, optional): State dictionary for traces, etc.
            **kwargs: Context information (e.g., 'phase', 'reward', 'dt')

        Returns:
            delta_w (Tensor or None): Weight change matrix (same shape as current_weights)
            logs (Dict): Diagnostic metrics (e.g., 'mean_ltp', 'loss')
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return configuration parameters for serialization."""
        return {}

# --- Alias for Backward Compatibility ---
# これにより、古いコードが 'BioLearningRule' をインポートしてもエラーにならない
BioLearningRule = PlasticityRule