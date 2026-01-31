# ファイルパス: snn_research/systems/autonomous_learning_loop.py
# 日本語タイトル: Autonomous Learning Loop v2.1 (Fixes for Sleep API)
# 目的・内容:
#   SleepConsolidator v2.2 (Hippocampusベース) に対応。
#   Hippocampusを初期化し、エピソード記憶を経由して睡眠学習を行うように修正。

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Any, Optional
import logging

from snn_research.systems.embodied_vlm_agent import EmbodiedVLMAgent
from snn_research.cognitive_architecture.intrinsic_motivation import (
    IntrinsicMotivationSystem,
)
from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator
from snn_research.cognitive_architecture.hippocampus import Hippocampus
from snn_research.cognitive_architecture.cortex import Cortex

logger = logging.getLogger(__name__)


class AutonomousLearningLoop(nn.Module):
    # [Fix] 引数に agent, optimizer, device を追加
    def __init__(
        self,
        config: Dict[str, Any],
        agent: EmbodiedVLMAgent,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.agent: EmbodiedVLMAgent = agent.to(device)  # type: ignore
        self.optimizer = optimizer

        logger.info(f"⚙️ Initializing AutonomousLearningLoop on {device}...")

        self.motivator: IntrinsicMotivationSystem = IntrinsicMotivationSystem(
            curiosity_weight=config.get("curiosity_weight", 1.0)
        )

        # [Fix] Add missing components
        import torch.optim as optim

        self.world_predictor = nn.Linear(512, 512).to(device)  # Dummy placeholder
        self.predictor_optimizer = optim.Adam(
            self.world_predictor.parameters(), lr=1e-3
        )

        # Initialize Hippocampus & SleepSystem
        self.hippocampus = Hippocampus(capacity=1000, input_dim=512, device=str(device))
        self.sleep_system = SleepConsolidator(substrate=self.hippocampus)

        # Energy System
        energy_capacity = config.get("energy_capacity", 1000.0)
        fatigue_threshold = config.get("fatigue_threshold", 800.0)

        self.energy = energy_capacity
        self.max_energy = energy_capacity
        self.fatigue = 0.0
        self.fatigue_threshold = fatigue_threshold

    def step(self, observation: torch.Tensor) -> Dict[str, Any]:
        # [Fix] Define variables from observation
        current_image = observation
        current_text: Optional[torch.Tensor] = None
        next_image: Optional[torch.Tensor] = None

        # 1. Sleep Check
        if self._should_sleep():
            return self._perform_sleep_cycle()

        self.agent.train()
        self.world_predictor.train()

        # 2. Perception & Action
        agent_out = self.agent(current_image, current_text)
        z_t = agent_out.get("fused_context")
        action = agent_out.get("action_pred")

        # 3. Prediction
        if z_t is not None and action is not None:
            if z_t.dim() > 2:
                z_t = z_t.mean(dim=1)
            pred_input = torch.cat([z_t, action], dim=-1)
            z_next_pred = self.world_predictor(pred_input)
        else:
            z_next_pred = torch.zeros(1, 512).to(self.device)

        # 4. Surprise Calculation
        surprise = 0.0
        prediction_loss = torch.tensor(0.0).to(self.device)

        if next_image is not None:
            with torch.no_grad():
                next_out = self.agent.vlm(next_image, current_text)
                z_next_actual = next_out.get("fused_representation")
                if z_next_actual is not None:
                    if z_next_actual.dim() > 2:
                        z_next_actual = z_next_actual.mean(dim=1)
                    prediction_loss = F.mse_loss(z_next_pred, z_next_actual)
                    surprise = torch.clamp(prediction_loss, 0.0, 1.0).item()

        # 5. Motivation
        self.motivator.process(input_payload=z_t, prediction_error=surprise)
        intrinsic_reward = self.motivator.calculate_intrinsic_reward(surprise=surprise)

        # 6. Memory Storage (Hippocampus)
        # 辞書形式のエピソードとして保存
        episode = {
            "input": current_image.detach().cpu(),  # CPUに退避してメモリ節約
            "text": current_text.detach().cpu() if current_text is not None else None,
            "reward": intrinsic_reward,
            "surprise": surprise,
        }
        self.hippocampus.process(episode)

        # 7. Learning
        total_loss = prediction_loss + agent_out.get("alignment_loss", 0) * 0.1
        self.optimizer.zero_grad()
        self.predictor_optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.predictor_optimizer.step()

        # 8. Homeostasis
        self.energy -= 1.0
        self.fatigue += 0.5 + surprise * 2.0

        drives = self.motivator.update_drives(
            surprise=surprise,
            energy_level=self.energy,
            fatigue_level=self.fatigue,
            task_success=True,
        )

        return {
            "mode": "wake",
            "loss": total_loss.item(),
            "surprise": surprise,
            "energy": self.energy,
            "drives": drives,
        }

    def _should_sleep(self) -> bool:
        if self.fatigue >= self.fatigue_threshold or self.energy <= 0:
            return True
        return False

    def _perform_sleep_cycle(self) -> Dict[str, Any]:
        # SleepConsolidator v2.2 API
        report = self.sleep_system.perform_sleep_cycle(duration_cycles=5)

        self.fatigue = 0.0
        self.energy = self.max_energy * 0.9

        return {"mode": "sleep", "report": report, "energy": self.energy}
