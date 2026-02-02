# ファイルパス: scripts/experiments/brain/run_stability_enhanced_learning.py
# 日本語タイトル: 安定性強化型局所学習実行スクリプト (恒常性維持導入版)
# 目的・内容:
#   初期発火の消失を防ぎ、安定性を Target > 95% に引き上げるための改善。
#   閾値の動的調整 (Homeostasis) を実装し、生物学的な安定化をシミュレートする。

import torch
import logging
from typing import Dict

# コアライブラリのインポート
from snn_research.core.neuromorphic_os import NeuromorphicOS
from snn_research.core.layers.lif_layer import LIFLayer
from snn_research.learning_rules.stdp import STDPRule
from snn_research.core.base import BaseModel

# Phase 2 完了条件および定数
STABILITY_TARGET = 0.95
SPIKE_RATE_TARGET = 0.05
TARGET_ACTIVITY = 0.02  # 目標とする平均発火率


class HomeostaticBrain(BaseModel):
    """
    恒常性維持機能を備えた SNN モデル。
    発火率を監視し、LIF レイヤーの閾値を動的に調整することで学習を安定化させる。
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        # 重みの初期値を少し大きく設定 (0.01 -> 0.1) して初期消失を防ぐ
        self.sensory_layer = LIFLayer(
            input_features=input_dim,
            neurons=hidden_dim,
            name="cortex_l4",
            threshold=2.6,
        )
        self.sensory_layer.W.data *= 10.0

        self.output_layer = LIFLayer(
            input_features=hidden_dim, neurons=10, name="cortex_l23"
        )

        # 局所学習則: STDP
        self.learning_rule = STDPRule(learning_rate=0.0005)

        # 恒常性パラメータ (Intrinsic Plasticity)
        self.threshold_min = 0.1
        self.threshold_max = 10.0
        self.homeostasis_rate = 0.5  # 調整速度

    def step(self, x: torch.Tensor, dopamine: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        時間ステップごとの更新と閾値の自己調整
        """
        # 1. 順伝播
        out1 = self.sensory_layer(x)
        spikes1 = out1["spikes"]

        # 2. 恒常性維持 (Homeostasis): 発火率に基づき閾値を調整
        # 発火が少なければ閾値を下げ、多すぎれば上げる
        current_rate = spikes1.mean().item()
        error = current_rate - TARGET_ACTIVITY

        # 閾値を動的に変更 (LIFLayer.threshold 属性を直接操作)
        new_threshold = self.sensory_layer.threshold + (self.homeostasis_rate * error)
        self.sensory_layer.threshold = max(
            self.threshold_min, min(self.threshold_max, new_threshold)
        )

        # 3. 局所学習 (STDP)
        delta_w, _ = self.learning_rule.update(
            pre_spikes=x,
            post_spikes=spikes1,
            current_weights=self.sensory_layer.W,
            dopamine_level=dopamine,
            local_state={},  # 簡易化のため
        )

        if delta_w is not None:
            # 安定化のため更新量をクリッピング
            self.sensory_layer.W.data += torch.clamp(delta_w, -0.01, 0.01)

        return {
            "hidden_spikes": spikes1,
            "current_threshold": self.sensory_layer.threshold,
        }


def run_experiment():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("StabilityResearch")

    input_dim = 784
    brain = HomeostaticBrain(input_dim=input_dim, hidden_dim=256)
    _ = NeuromorphicOS(brain=brain)

    logger.info("Starting Stability Validation with Homeostasis Integration...")

    stability_count = 0
    total_steps = 1000

    for t in range(total_steps):
        # 信号強度を高めるため、入力スパイクの密度を調整
        inputs = (torch.rand(1, input_dim) > 0.90).float()

        results = brain.step(inputs)
        activity = results["hidden_spikes"].mean().item()

        # 安定性の判定: 極端な沈黙または暴走を回避しているか
        if 0.005 < activity < SPIKE_RATE_TARGET:
            stability_count += 1

        if t % 100 == 0:
            current_stability = stability_count / (t + 1)
            logger.info(
                f"Step {t:04d}: Stability={current_stability:.1%}, "
                f"Activity={activity:.3%}, Threshold={results['current_threshold']:.3f}"
            )

    final_stability = stability_count / total_steps
    logger.info(f"Final Stability Result: {final_stability:.2%}")

    if final_stability >= STABILITY_TARGET:
        logger.info("✅ Phase 2 Success Criteria Met: Stability is stable.")
    else:
        logger.warning(
            "⚠️ Stability target not reached. Consider increasing homeostasis_rate."
        )


if __name__ == "__main__":
    run_experiment()
