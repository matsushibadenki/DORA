# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: Artificial Brain Integrated v17.0 (Biological Time Aware)
# 目的・内容:
#   - process_tick()の実装による、時間経過に伴う代謝と自発的思考。
#   - reset_state()によるメモリリーク防止機能の維持。

import torch
import torch.nn as nn
import logging
import random
from typing import Dict, Any, Optional, Union, List

# --- Core Architectures ---
from snn_research.core.networks.bio_pc_network import BioPCNetwork
from snn_research.models.transformer.sformer import ScaleAndFireTransformer

# --- Cognitive Components ---
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
from snn_research.cognitive_architecture.prefrontal_cortex import PrefrontalCortex
from snn_research.cognitive_architecture.basal_ganglia import BasalGanglia
from snn_research.cognitive_architecture.intrinsic_motivation import (
    IntrinsicMotivationSystem,
)
from snn_research.cognitive_architecture.rag_snn import RAGSystem
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator
from snn_research.cognitive_architecture.hippocampus import Hippocampus
from snn_research.cognitive_architecture.motor_cortex import MotorCortex

# Optional: High-level reasoning
try:
    from snn_research.cognitive_architecture.theory_of_mind import TheoryOfMind
    from snn_research.cognitive_architecture.causal_inference_engine import (
        CausalInferenceEngine,
    )

    ADVANCED_MODULES_AVAILABLE = True
except ImportError:
    ADVANCED_MODULES_AVAILABLE = False

logger = logging.getLogger(__name__)


class ArtificialBrain(nn.Module):
    """
    Artificial Brain v17.0
    Supports biological time ticking and metabolic energy consumption.
    """

    def __init__(self, config: Dict[str, Any], device_name: Optional[str] = None):
        super().__init__()
        self.config = config

        # [Fix] Handle 'auto' device properly
        target_device = device_name
        if not target_device or target_device == "auto" or str(target_device) == "None":
            target_device = config.get("device")

        if not target_device or target_device == "auto" or str(target_device) == "None":
            if torch.cuda.is_available():
                target_device = "cuda"
            elif torch.backends.mps.is_available():
                target_device = "mps"
            else:
                target_device = "cpu"

        self.device = torch.device(target_device)

        self.is_awake = True
        self.plasticity_enabled = False
        self.sleep_cycle_count = 0
        self.d_model = config.get("model", {}).get("d_model", 256)
        
        # 時間経過によるパラメータ
        self.boredom_counter = 0.0
        self.boredom_threshold = 10.0 # 秒数：これ以上入力がないと自発的思考を開始

        # 1. 物理層
        self._init_core_brain()

        self.astrocyte = AstrocyteNetwork(
            max_energy=1000.0, fatigue_threshold=80.0, device=str(self.device)
        )

        # 2. 認知層
        self.workspace = GlobalWorkspace(
            dim=self.d_model, decay=config.get("workspace_decay", 0.9)
        )

        self.motivation_system = IntrinsicMotivationSystem(
            curiosity_weight=config.get("curiosity_weight", 1.0), boredom_threshold=0.8
        )

        self.rag_system = RAGSystem(
            embedding_dim=self.d_model,
            vector_store_path=config.get("memory_path", "./runtime_state/memory"),
        )

        self.hippocampus = Hippocampus(
            capacity=200,
            input_dim=self.d_model if hasattr(self, "d_model") else 784,
            device=str(self.device),
        )

        self.pfc = PrefrontalCortex(
            workspace=self.workspace,
            motivation_system=self.motivation_system,
            d_model=self.d_model,
            device=str(self.device),
        )

        self.basal_ganglia = BasalGanglia(
            workspace=self.workspace, selection_threshold=0.4
        )

        self.motor_cortex = MotorCortex(
            actuators=["default_actuator"], device=str(self.device)
        )

        self.theory_of_mind: Optional[TheoryOfMind] = None
        self.causal_engine: Optional[CausalInferenceEngine] = None

        if ADVANCED_MODULES_AVAILABLE and config.get("enable_advanced_cognition", True):
            self.theory_of_mind = TheoryOfMind(self.workspace, self.rag_system)
            self.causal_engine = CausalInferenceEngine(self.rag_system, self.workspace)

        # 3. メンテナンス層
        self.sleep_manager = SleepConsolidator(substrate=self.core)

        logger.info(f"🚀 Artificial Brain v17.0 initialized on {self.device}.")

    @property
    def cycle_count(self) -> int:
        return self.sleep_cycle_count

    @property
    def device_name(self) -> str:
        return str(self.device)

    # --- New Biological Methods ---

    def process_tick(self, delta_time: float):
        """
        時間経過（Tick）を処理する。入力がないアイドル時にOSから呼ばれる。
        1. エネルギー消費（基礎代謝）
        2. 退屈（Boredom）の更新と自発的思考
        """
        if not self.is_awake:
            return

        # 1. 基礎代謝 (Time-based Metabolism)
        # 1秒あたり1.0エネルギーを消費すると仮定
        metabolic_rate = 1.0 
        energy_cost = metabolic_rate * delta_time
        
        # Astrocyteのエネルギーを直接減らす（メソッドがない場合のフォールバック）
        if hasattr(self.astrocyte, "consume_energy"):
             self.astrocyte.consume_energy(energy_cost)
        else:
             # 直接属性操作
             self.astrocyte.current_energy = max(0.0, self.astrocyte.current_energy - energy_cost)
        
        # 2. 退屈の更新
        self.boredom_counter += delta_time
        
        # 3. 自発的思考 (Internal Monologue) のトリガー
        if self.boredom_counter > self.boredom_threshold:
            self._trigger_spontaneous_thought()
            self.boredom_counter = 0.0 # リセット

    def _trigger_spontaneous_thought(self):
        """退屈時に呼び出され、独り言や過去の回想を行う"""
        logger.info("💭 Brain is bored. Triggering spontaneous thought...")
        
        # ランダムなトピックまたは状態報告
        topics = [
            "静かですね。",
            f"お腹が空いてきました... (Energy: {self.astrocyte.current_energy:.1f})",
            "新しいことを学びたいです。",
            "システムログを整理しています..."
        ]
        thought = random.choice(topics)
        
        # 自身に入力として与える（思考ループ）
        self.process_step(f"[Internal Monologue] {thought}")


    # --- Existing Methods ---

    def run_cycle(self, sensory_input: Any, phase: str = "wake") -> Dict[str, Any]:
        """Legacy wrapper"""
        return self.process_step(sensory_input)

    def _init_core_brain(self):
        model_conf = self.config.get("model", {})
        arch_type = model_conf.get("architecture_type", "sformer")

        if arch_type == "bio_pc_network":
            layer_sizes = model_conf.get("network", {}).get(
                "layer_sizes", [784, 512, 256, self.d_model]
            )
            self.core = BioPCNetwork(
                input_shape=(784,), layer_sizes=layer_sizes, config=self.config
            )
        elif arch_type == "sformer":
            self.core = ScaleAndFireTransformer(
                vocab_size=self.config.get("data", {}).get("vocab_size", 50257),
                d_model=self.d_model,
                num_layers=model_conf.get("num_layers", 6),
            )
        else:
            self.core = BioPCNetwork(
                input_shape=(784,),
                layer_sizes=[64, 128, self.d_model],
                config=self.config,
            )

        self.core.to(self.device)
        self.thinking_engine = self.core

    @property
    def cortex(self):
        return self

    def get_all_knowledge(self) -> List[str]:
        if hasattr(self.rag_system, "knowledge_base"):
            return self.rag_system.knowledge_base
        return []

    def reset_state(self):
        """
        全モジュールの内部状態（電位、キャッシュ）をリセットし、計算グラフを切断する。
        """
        def _recursive_reset(module):
            if hasattr(module, "reset_state") and callable(module.reset_state):
                module.reset_state()
            elif (
                hasattr(module, "reset")
                and callable(module.reset)
                and module is not self
            ):
                module.reset()

        self.core.apply(_recursive_reset)

        if hasattr(self.workspace, "reset"):
            self.workspace.reset()

        if hasattr(self.hippocampus, "reset_state"):
            self.hippocampus.reset_state()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()

    def process_step(
        self, sensory_input: Union[torch.Tensor, str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        
        # 入力があったので退屈カウンターをリセット
        self.boredom_counter = 0.0

        if not self.is_awake:
            return {"status": "sleeping", "output": None}

        if self.astrocyte.current_energy <= 5.0:
            logger.warning("⚠️ Brain Energy Critically Low! Entering low-power mode.")
            return {"status": "fatigued", "output": None}

        perception_embedding = None
        perception_content = {}
        output_tensor = None

        if isinstance(sensory_input, torch.Tensor):
            sensory_input = sensory_input.to(self.device)
            model_output = self.core(sensory_input)

            if isinstance(model_output, tuple):
                features = model_output[0]
            elif isinstance(model_output, dict):
                if "states" in model_output:
                    features = model_output["states"]
                elif "fused_representation" in model_output:
                    features = model_output["fused_representation"]
                elif "last_hidden_state" in model_output:
                    features = model_output["last_hidden_state"]
                else:
                    features = next(
                        (
                            v
                            for v in model_output.values()
                            if isinstance(v, torch.Tensor)
                        ),
                        torch.zeros_like(sensory_input),
                    )
            else:
                features = model_output

            output_tensor = features
            perception_embedding = features
            perception_content = {"features": features, "modality": "visual"}

            if hasattr(self, "hippocampus"):
                if isinstance(features, torch.Tensor):
                    self.hippocampus.store_episode(features.detach())

        elif isinstance(sensory_input, str):
            perception_content = {"text": sensory_input, "modality": "language"}
            if (
                self.rag_system.embeddings is not None
                and len(self.rag_system.embeddings) > 0
            ):
                perception_embedding = self.rag_system._encode([sensory_input]).to(
                    self.device
                )
                if perception_embedding.shape[-1] != self.d_model:
                    perception_embedding = torch.nn.functional.pad(
                        perception_embedding,
                        (0, self.d_model - perception_embedding.shape[-1]),
                    )
                perception_content["features"] = perception_embedding
                output_tensor = perception_embedding

        if perception_embedding is not None:
            self.workspace.upload_to_workspace(
                source_name="sensory_cortex", content=perception_content, salience=0.8
            )

        self.workspace.step()
        conscious_content = self.workspace.get_current_content()

        surprise = 0.1
        current_drives = self.motivation_system.process(
            sensory_input, prediction_error=surprise
        )

        firing_rate = 0.1
        if hasattr(self.core, "get_mean_firing_rate"):
            firing_rate = self.core.get_mean_firing_rate()
        self.astrocyte.monitor_neural_activity(firing_rate)

        pfc_plan = self.pfc.plan(conscious_content)
        action_candidates = []
        if pfc_plan:
            action_candidates.append(
                {
                    "action": pfc_plan.get("directive", "wait"),
                    "value": pfc_plan.get("priority", 0.5),
                    "source": "pfc",
                }
            )

        selected_action = self.basal_ganglia.select_action(
            external_candidates=action_candidates, emotion_context=current_drives
        )
        
        # 思考したコストとしてエネルギー消費 (思考1回あたり2.0消費)
        self.astrocyte.consume_energy(2.0)

        return {
            "output": output_tensor,
            "perception": perception_content.get("text", "tensor_data"),
            "conscious_broadcast": conscious_content,
            "pfc_goal": self.pfc.current_goal,
            "drives": current_drives,
            "action": selected_action,
            "energy": self.astrocyte.current_energy,
            "executed_modules": [
                "Core",
                "GlobalWorkspace",
                "PFC",
                "BasalGanglia",
                "MotorCortex",
            ],
        }

    def run_cognitive_cycle(
        self, text_or_input: Union[str, torch.Tensor]
    ) -> Dict[str, Any]:
        result = self.process_step(text_or_input)
        return {
            "output": result.get("output"),
            "executed_modules": result.get("executed_modules", []),
            "consciousness": str(
                result.get("conscious_broadcast", {}).get("source", "None")
            ),
            "thought": str(text_or_input),
        }

    def sleep_cycle(self):
        self.sleep()
        self.wake_up()

    def retrieve_knowledge(self, query: str) -> List[str]:
        return self.rag_system.search(query)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.process_step(x)
        out = result.get("output")
        if out is None:
            return torch.zeros(1, self.d_model, device=self.device)
        return out

    def sleep(self):
        logger.info(">>> 💤 Sleep Cycle Initiated <<<")
        self.is_awake = False
        self.set_plasticity(False)
        self.sleep_cycle_count += 1

        self.reset_state()

        self.astrocyte.replenish_energy(500.0)
        self.astrocyte.clear_fatigue(50.0)

        stats = self.sleep_manager.perform_maintenance(self.sleep_cycle_count)
        logger.info(f"Sleep Maintenance Report: {stats}")

    def wake_up(self):
        logger.info(">>> 🌅 Wake Up <<<")
        self.is_awake = True
        self.set_plasticity(True)

    def set_plasticity(self, active: bool):
        self.plasticity_enabled = active
        if hasattr(self.core, "set_online_learning"):
            self.core.set_online_learning(active)
        else:
            self.core.train(active)

    def get_brain_status(self) -> Dict[str, Any]:
        current_thought = self.workspace.get_current_thought()
        thought_str = (
            str(current_thought.tolist())[:50]
            if current_thought is not None
            else "None"
        )

        return {
            "state": "AWAKE" if self.is_awake else "SLEEPING",
            "cycle": self.sleep_cycle_count,
            "energy": self.astrocyte.current_energy,
            "fatigue": self.astrocyte.fatigue,
            "current_goal": self.pfc.current_goal,
            "drives": self.motivation_system.get_internal_state(),
            "conscious_content": thought_str,
        }