# ファイルパス: snn_research/learning_rules/active_inference.py
# 日本語タイトル: Active Inference Plasticity Rule
# 目的・内容:
#   Fristonの自由エネルギー原理に基づく局所学習則。
#   シナプス重みは「予測誤差」を最小化するように更新される。
#   Delta W = - learning_rate * Prediction_Error * Pre_Activity

import torch
import logging
from typing import Dict, Any, Tuple, Optional

from snn_research.learning_rules.base_rule import PlasticityRule

logger = logging.getLogger(__name__)

class ActiveInferenceRule(PlasticityRule):
    """
    Active Inference (Predictive Coding) Learning Rule.
    
    The brain minimizes Free Energy (Prediction Error).
    Synaptic update:
      dWe / dt  = epsilon * (error * pre_activity)
    
    Where:
      error = (Sensory_Input - Prediction)
      Prediction = W * Pre_Activity
    """

    def __init__(
        self,
        learning_rate: float = 0.005,
        decay_rate: float = 0.01
    ):
        self.lr = learning_rate
        self.decay = decay_rate

    def update(
        self, 
        pre_spikes: torch.Tensor, 
        post_spikes: torch.Tensor, 
        current_weights: torch.Tensor, 
        local_state: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """
        Args:
            pre_spikes: Top-down prediction source (or lateral)
            post_spikes: Current activity (representing Prediction Error in some architectures)
            kwargs: Must contain 'target_activity' or 'sensory_input' if post is error unit
        """
        # Active Inferenceの実装はアーキテクチャに強く依存するが、
        # ここでは「ポストシナプスニューロンが予測信号を出力しようとしている」と仮定し、
        # その予測と実際のターゲットとの誤差を使う汎用的な形式とする。
        
        target = kwargs.get("target_signal")
        
        if target is None:
            # ターゲットがない場合（自律モード）、STDP的なHeuristicにフォールバックするか、更新しない
            return None, {}

        # 予測 (Current Prediction)
        # 本来は膜電位(Membrane)を使うのが正確だが、ここではスパイクレートで近似
        # Prediction = Weights * Pre
        # (Batch, N_post)
        prediction_approx = torch.matmul(pre_spikes, current_weights.t())
        
        # 予測誤差 (Prediction Error)
        error = target - prediction_approx
        
        # 重み更新: Error * Pre (誤差を減らす方向へ)
        # dW = lr * (Error^T * Pre) 
        delta_w = self.lr * torch.matmul(error.t(), pre_spikes)
        
        # Weight Decay
        delta_w -= self.decay * current_weights

        logs = {
            "mean_prediction_error": error.abs().mean().item(),
            "mean_prediction": prediction_approx.mean().item()
        }

        return delta_w, logs

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "ActiveInference",
            "lr": self.lr
        }