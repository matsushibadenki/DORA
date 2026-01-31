# ファイルパス: snn_research/learning_rules/bcm_rule.py
# Title: BCM Learning Rule (Fixed)
# Description:
#   メソッドシグネチャを基底クラス (PlasticityRule) に準拠。

from __future__ import annotations

import torch
from typing import Dict, Any, Optional, Tuple, cast
from .base_rule import BioLearningRule


class BCMLearningRule(BioLearningRule):
    """
    BCM (Bienenstock-Cooper-Munro) 学習規則。
    """
    avg_post_activity: torch.Tensor

    def __init__(
        self,
        learning_rate: float = 0.005,
        tau_avg: float = 500.0,
        target_rate: float = 0.01,
        dt: float = 1.0
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.tau_avg = max(1.0, tau_avg)
        self.target_rate = target_rate
        self.dt = dt

        # mypy用に初期化。実際は_initialize_tracesで設定されるか、register_bufferを使うべき。
        # ここではNone許容ではなくTensor型として扱い、実行時にチェックする。
        self.register_buffer('avg_post_activity', torch.zeros(1))
        self.avg_decay_factor = dt / self.tau_avg
        self.stability_eps = 1e-6
        
        # 初期化フラグ
        self._initialized = False

        print(f"🧠 BCM V16.5 initialized (Target: {target_rate}, High Stability Mode)")

    def _initialize_traces(self, post_shape: int, device: torch.device) -> None:
        self.avg_post_activity = torch.full(
            (post_shape,), self.target_rate, device=device)
        self._initialized = True

    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        current_weights: torch.Tensor,
        **kwargs: Any
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:

        # バッチ平均
        pre_avg = pre_spikes.mean(dim=0) if pre_spikes.dim() > 1 else pre_spikes
        post_avg = post_spikes.mean(dim=0) if post_spikes.dim() > 1 else post_spikes

        if not self._initialized or self.avg_post_activity.shape[0] != post_avg.shape[0]:
            self._initialize_traces(post_avg.shape[0], post_spikes.device)

        avg_act = self.avg_post_activity

        # 1. 閾値 (theta) の動的更新
        with torch.no_grad():
            new_avg = (1.0 - self.avg_decay_factor) * avg_act + self.avg_decay_factor * post_avg
            self.avg_post_activity = new_avg.detach()

        # 2. 閾値関数の計算
        theta = (avg_act ** 2) / (self.target_rate + self.stability_eps)

        # 3. 状態遷移関数: post * (post - theta)
        phi = post_avg * (post_avg - theta)

        # 4. 重み更新量
        dw = self.learning_rate * torch.outer(phi, pre_avg)

        # 5. ログデータ
        logs = {
            "mean_theta": theta.mean().item(),
            "mean_phi": phi.mean().item()
        }

        return dw, logs