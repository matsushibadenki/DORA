# snn_research/systems/autonomous_learning_loop.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional

from snn_research.systems.embodied_vlm_agent import EmbodiedVLMAgent
from snn_research.adaptive.intrinsic_motivator import IntrinsicMotivator
from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator
from snn_research.cognitive_architecture.hippocampus import Hippocampus

logger = logging.getLogger(__name__)

class AutonomousLearningLoop(nn.Module):
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
        self.agent = agent.to(device)
        self.optimizer = optimizer

        logger.info(f"⚙️ Initializing AutonomousLearningLoop on {device}...")

        # 動作確認済みのMotivatorを使用
        self.motivator = IntrinsicMotivator(config)
        self.motivator.to(device)

        # 世界モデル（予測器）
        self.world_predictor = nn.Linear(512 + 64, 512).to(device)
        self.predictor_optimizer = torch.optim.Adam(
            self.world_predictor.parameters(), lr=1e-3
        )

        # 記憶システムの初期化
        self.hippocampus = Hippocampus(capacity=1000, input_dim=512, device=str(device))
        
        # [修正ポイント 1] SleepConsolidatorの対象(substrate)をAgentに変更
        # Hippocampusはパラメータを持たないため、学習対象にはなり得ない
        self.sleep_system = SleepConsolidator(substrate=self.agent)

        # エネルギーシステム
        self.energy_capacity = config.get("energy_capacity", 1000.0)
        self.fatigue_threshold = config.get("fatigue_threshold", 800.0)
        self.energy = self.energy_capacity
        self.fatigue = 0.0

    def step(self, current_image: torch.Tensor, current_text: Optional[torch.Tensor] = None, next_image: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        ライフサイクルの1ステップを実行
        """
        # 1. Sleep Check
        if self._should_sleep():
            return self._perform_sleep_cycle()

        self.agent.train()
        self.world_predictor.train()

        # 2. Perception & Action
        agent_out = self.agent(current_image, current_text)
        
        z_t = agent_out.get("fused_context")
        if z_t is None: z_t = torch.zeros(current_image.shape[0], 512).to(self.device)
            
        action = agent_out.get("action_pred")
        if action is None: action = torch.zeros(current_image.shape[0], 64).to(self.device)
        
        # 3. Prediction (世界モデル)
        if z_t.dim() > 2: z_t = z_t.mean(dim=1)
        if action.shape[-1] != 64:
             action = torch.zeros(action.shape[0], 64).to(self.device)

        pred_input = torch.cat([z_t, action], dim=-1)
        z_next_pred = self.world_predictor(pred_input)

        # 4. Surprise Calculation
        surprise = 0.0
        prediction_loss = torch.tensor(0.0).to(self.device)

        z_next_actual = None # 初期化
        if next_image is not None:
            with torch.no_grad():
                if hasattr(self.agent, "vlm"):
                    next_out = self.agent.vlm(next_image, current_text)
                    if isinstance(next_out, dict):
                        z_next_actual = next_out.get("fused_representation")
                    else:
                         z_next_actual = next_out
                else:
                    z_next_actual = z_t
                
                if z_next_actual is not None:
                    if z_next_actual.dim() > 2: z_next_actual = z_next_actual.mean(dim=1)
                    if z_next_actual.shape == z_next_pred.shape:
                        prediction_loss = F.mse_loss(z_next_pred, z_next_actual)
                        surprise = torch.clamp(prediction_loss, 0.0, 1.0).item()

        # 5. Motivation
        z_next_target = z_next_actual if z_next_actual is not None else z_next_pred
        intrinsic_reward = self.motivator.compute_reward(z_next_pred, z_next_target)

        # 6. Memory Storage
        episode = {
            "input": "Visual Input", # デモ用の簡易テキスト
            "text": "Context",
            "reward": intrinsic_reward.item(),
            "surprise": surprise,
            # 実際のテンソルを保存すると重いが、デモではそのまま使う
            # "tensor_data": current_image.detach().cpu() 
        }
        self.hippocampus.process(episode)
        
        # [修正ポイント 2] 睡眠システムにもエピソードを供給 (夢を見るため)
        self.sleep_system.add_episode(episode)

        # 7. Learning
        total_loss = prediction_loss + agent_out.get("alignment_loss", torch.tensor(0.0).to(self.device)) * 0.1
        
        self.optimizer.zero_grad()
        self.predictor_optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.predictor_optimizer.step()

        # 8. Homeostasis
        self.energy -= 5.0 # デモ用に早く減るように調整
        self.fatigue += 1.0 + surprise * 5.0

        return {
            "mode": "wake",
            "loss": total_loss.item(),
            "surprise": surprise,
            "energy": self.energy,
            "intrinsic_reward": intrinsic_reward.item(),
            "fatigue": self.fatigue
        }

    def _should_sleep(self) -> bool:
        if self.fatigue >= self.fatigue_threshold or self.energy <= 0:
            return True
        return False

    def _perform_sleep_cycle(self) -> Dict[str, Any]:
        logger.info("💤 Entering Sleep Cycle...")
        
        # SleepConsolidator API
        report = self.sleep_system.perform_sleep_cycle(duration_cycles=5)

        self.fatigue = 0.0
        self.energy = self.energy_capacity * 1.0
        logger.info("🌅 Waking up refreshed!")

        return {"mode": "sleep", "sleep_loss": 0.1, "report": report, "energy": self.energy}