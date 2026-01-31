# ファイルパス: snn_research/cognitive_architecture/intrinsic_motivation.py
# 日本語タイトル: Intrinsic Motivation System v2.6 (Update State Support)
# 目的・内容:
#   - update_state メソッドを追加し、外部報酬による動機更新を可能にする。

import torch.nn as nn
import logging
import numpy as np
from typing import Dict, Any, Optional, Callable, List

logger = logging.getLogger(__name__)

KnowledgeCallback = Callable[[str, str, float, str], None]


class IntrinsicMotivationSystem(nn.Module):
    def __init__(
        self,
        curiosity_weight: float = 1.0,
        boredom_decay: float = 0.995,
        boredom_threshold: float = 0.8,
        novelty_bonus: float = 1.0,
        competence_bonus: float = 0.5,
    ):
        super().__init__()
        self.curiosity_weight = curiosity_weight
        self.boredom_decay = boredom_decay
        self.boredom_threshold = boredom_threshold
        self.novelty_bonus = novelty_bonus
        self.competence_bonus = competence_bonus

        self.last_input_hash: Optional[int] = None
        self.repetition_count = 0

        self.drives: Dict[str, float] = {
            "curiosity": 0.5,
            "boredom": 0.0,
            "survival": 1.0,
            "comfort": 0.5,
            "competence": 0.3,
            "fear": 0.0,  # デモ用にfearを初期化
        }

        self._knowledge_callbacks: List[KnowledgeCallback] = []
        logger.info("🔥 Intrinsic Motivation System v2.6 initialized.")

    def register_knowledge_callback(self, callback: KnowledgeCallback) -> None:
        self._knowledge_callbacks.append(callback)

    def process(
        self, input_payload: Any, prediction_error: Optional[float] = None
    ) -> Dict[str, float]:
        surprise = 0.0
        if prediction_error is not None:
            surprise = min(1.0, prediction_error)
            if surprise < 0.1:
                self.repetition_count += 1
                boredom_delta = 0.05 * self.repetition_count
                self._update_drive("competence", 0.05)
            else:
                self.repetition_count = 0
                boredom_delta = -0.2
                self._update_drive("competence", -0.02)
        elif isinstance(input_payload, (str, int, float)):
            input_hash = hash(input_payload)
            if input_hash == self.last_input_hash:
                self.repetition_count += 1
                surprise = 0.0
                boredom_delta = 0.1 * self.repetition_count
            else:
                self.repetition_count = 0
                surprise = 1.0
                boredom_delta = -0.5
            self.last_input_hash = input_hash
        else:
            boredom_delta = 0.01

        self._update_drive("curiosity", surprise * 0.2 - 0.01)
        self.drives["boredom"] = float(
            np.clip(self.drives["boredom"] + boredom_delta, 0.0, 1.0)
        )
        return self.get_internal_state()

    def update_state(self, state_update: Dict[str, float]) -> None:
        """
        [New] 外部からの状態更新（報酬など）を適用する。
        """
        # 報酬処理
        if "reward" in state_update:
            reward = state_update.pop("reward")
            if reward > 0:
                self._update_drive("competence", 0.1)
                self._update_drive("comfort", 0.1)
                self._update_drive("fear", -0.1)
            else:
                # 罰（負の報酬）
                self._update_drive("fear", 0.2)
                self._update_drive("comfort", -0.2)
                self._update_drive("survival", -0.05)

        # その他の状態を直接更新
        for key, value in state_update.items():
            self._update_drive(key, value)

    def _update_drive(self, key: str, delta: float):
        if key in self.drives:
            self.drives[key] = float(np.clip(self.drives[key] + delta, 0.0, 1.0))

    def calculate_intrinsic_reward(self, surprise: float) -> float:
        """
        [New] Calculate intrinsic reward based on surprise.
        """
        # Simple implementation
        return surprise * self.curiosity_weight

    def update_drives(
        self,
        surprise: float,
        energy_level: float,
        fatigue_level: float,
        task_success: bool,
    ) -> Dict[str, float]:
        """
        [New] Update drives based on various factors.
        """
        self._update_drive("curiosity", surprise * 0.1)
        # Assuming energy/fatigue affect comfort/survival
        if energy_level < 200:
            self._update_drive("survival", 0.1)
        if fatigue_level > 500:
            self._update_drive("comfort", -0.1)

        if task_success:
            self._update_drive("competence", 0.05)

        return self.get_internal_state()

    def get_internal_state(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.drives.items()}
