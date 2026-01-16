# snn_research/learning_rules/base_rule.py
# 修正: BioLearningRuleエイリアスの追加

from abc import ABC, abstractmethod
import torch
from typing import Any, Tuple, Optional, Dict

class PlasticityRule(ABC):
    """
    局所可塑性（Local Plasticity）を定義する抽象基底クラス。
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
        シナプス重みの更新量を計算する。
        
        Args:
            pre_spikes (Tensor): (Batch, N_pre)
            post_spikes (Tensor): (Batch, N_post)
            current_weights (Tensor): (N_post, N_pre)
            local_state (Dict): 局所状態
            **kwargs: その他

        Returns:
            delta_w (Tensor | None): 重み更新量
            logs (Dict): ログ
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        return {}

# --- Alias for Backward Compatibility ---
BioLearningRule = PlasticityRule