# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: Artificial Brain v20.2 (Attribute Fix)
# 目的: AttributeError 'cycle_count' を修正し、プロパティ定義を確実にする。

import torch
import torch.nn as nn
import logging
import os
import json
from typing import Dict, Any, Optional, Union, List

# --- Core Architectures ---
from snn_research.core.networks.bio_pc_network import BioPCNetwork
from snn_research.models.transformer.sformer import ScaleAndFireTransformer
from snn_research.core.snn_core import SpikingNeuralSubstrate

# --- Cognitive Components ---
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
from snn_research.cognitive_architecture.prefrontal_cortex import PrefrontalCortex
from snn_research.cognitive_architecture.basal_ganglia import BasalGanglia
from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem
from snn_research.cognitive_architecture.rag_snn import RAGSystem
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator
from snn_research.cognitive_architecture.hippocampus import Hippocampus
from snn_research.cognitive_architecture.motor_cortex import MotorCortex

logger = logging.getLogger(__name__)

class ArtificialBrain(nn.Module):
    def __init__(self, config: Dict[str, Any], device_name: Optional[str] = None, **kwargs):
        super().__init__()
        self.config = config

        # --- Device Setup ---
        target_device_str = device_name
        if not target_device_str or str(target_device_str) == "None":
            target_device_str = config.get("device", "cpu")
        
        # 'auto' Resolution
        if target_device_str == "auto":
            if torch.cuda.is_available():
                target_device_str = "cuda"
            elif torch.backends.mps.is_available():
                target_device_str = "mps"
            else:
                target_device_str = "cpu"
        
        # Kernel Mode Check
        training_conf = config.get("training", {})
        self.use_kernel = "event_driven" in str(training_conf.get("paradigm", ""))
        
        if self.use_kernel:
            target_device_str = "cpu"
            logger.info("⚡ [Brain] Kernel Mode Detected (CPU Optimized).")

        # Create device object
        try:
            self.device = torch.device(target_device_str) # type: ignore
        except Exception as e:
            logger.warning(f"⚠️ Invalid device '{target_device_str}' specified. Falling back to CPU. Error: {e}")
            self.device = torch.device("cpu")

        # --- Parameters ---
        self.is_awake = True
        self.sleep_cycle_count = 0
        self.d_model = config.get("model", {}).get("d_model", 256)
        self.boredom_counter = 0.0
        self.boredom_threshold = 10.0
        
        # --- Core Model ---
        self._init_core_brain_torch()
        self.kernel_substrate: Optional[SpikingNeuralSubstrate] = None
        if self.use_kernel:
            self._compile_to_kernel()

        # --- Components (Dependency Injection) ---
        self.astrocyte = kwargs.get("astrocyte_network") or AstrocyteNetwork(max_energy=1000.0, device=str(self.device))
        self.workspace = kwargs.get("global_workspace") or GlobalWorkspace(dim=self.d_model, decay=config.get("workspace_decay", 0.9))
        self.motivation_system = IntrinsicMotivationSystem(curiosity_weight=config.get("curiosity_weight", 1.0))
        
        # RAG System
        rag_path = os.path.join(config.get("runtime_dir", "./runtime_state"), "memory")
        self.rag_system = RAGSystem(embedding_dim=self.d_model, vector_store_path=rag_path)
        
        self.hippocampus = kwargs.get("hippocampus") or Hippocampus(capacity=200, input_dim=self.d_model, device=str(self.device))
        self.pfc = PrefrontalCortex(workspace=self.workspace, motivation_system=self.motivation_system, d_model=self.d_model, device=str(self.device))
        self.basal_ganglia = BasalGanglia(workspace=self.workspace, selection_threshold=0.3)
        self.motor_cortex = MotorCortex(actuators=["default_actuator"], device=str(self.device))

        # Sleep Manager
        self.sleep_manager = kwargs.get("sleep_consolidator") or SleepConsolidator(substrate=self.core_torch)

        logger.info(f"🚀 Artificial Brain v20.2 initialized on {self.device}.")

    @property
    def cycle_count(self) -> int:
        """睡眠サイクルの回数を返す"""
        return self.sleep_cycle_count

    @property
    def state(self) -> str:
        return "AWAKE" if self.is_awake else "SLEEPING"

    def _init_core_brain_torch(self):
        input_dim = 128
        hidden_dim = 512
        output_dim = self.d_model
        
        self.core_torch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.core_torch.to(self.device)

    def _compile_to_kernel(self):
        self.kernel_substrate = SpikingNeuralSubstrate(self.config, device="cpu")
        self.kernel_substrate.kernel_compiled = True

    def process_step(self, sensory_input: Union[torch.Tensor, str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        1回の認知サイクルを実行する。
        """
        self.boredom_counter = 0.0
        
        if not self.is_awake:
            return {"status": "dreaming", "energy": self.astrocyte.current_energy}

        # 1. Perception & RAG
        input_text = str(sensory_input)
        if isinstance(sensory_input, dict) and "text" in sensory_input:
            input_text = sensory_input["text"]
            
        retrieved_knowledge = self.rag_system.search(input_text, k=2)
        
        if retrieved_knowledge:
            self.workspace.upload_to_workspace(
                source_name="long_term_memory",
                content={"text": " ".join(retrieved_knowledge), "type": "context"},
                salience=0.6
            )

        # 2. Neural Processing
        output_tensor = torch.zeros(1, self.d_model, device=self.device)
        
        # 3. Workspace
        self.workspace.step()
        conscious_content = self.workspace.get_current_content()
        
        # 4. PFC & Action
        pfc_plan = self.pfc.plan(conscious_content)
        
        current_drives = self.motivation_system.get_internal_state()
        selected_action = self.basal_ganglia.select_action(
            external_candidates=[{"action": pfc_plan.get("directive"), "value": 0.5}] if pfc_plan else [],
            emotion_context=current_drives
        )
        
        # 5. Create Episode for SDFT
        if pfc_plan and pfc_plan.get("target"):
            episode = {
                "input": input_text,
                "thought_chain": f"Goal: {pfc_plan['goal']} -> Plan: {pfc_plan['directive']}",
                "answer": str(selected_action),
                # [Fix] Use self.cycle_count (property)
                "timestamp": self.cycle_count
            }
            self.sleep_manager.add_episode(episode)

        self.astrocyte.consume_energy(2.0)

        return {
            "output": output_tensor,
            "conscious_broadcast": conscious_content,
            "pfc_goal": self.pfc.current_goal,
            "retrieved_context": retrieved_knowledge,
            "action": selected_action,
            "energy": self.astrocyte.current_energy
        }

    # --- Persistence Methods ---

    def save_checkpoint(self, path: str):
        logger.info(f"💾 Saving brain state to {path}...")
        
        brain_state = {
            "model_state_dict": self.core_torch.state_dict(),
            "astrocyte": {
                "energy": self.astrocyte.current_energy,
                "fatigue": self.astrocyte.fatigue
            },
            "pfc": {
                "current_goal": self.pfc.current_goal
            },
            "episodic_buffer": self.sleep_manager.episodic_buffer,
            "sleep_cycle_count": self.sleep_cycle_count,
            "config": self.config
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(brain_state, path)
        self.rag_system.save()
        logger.info("✅ Brain state & Memories saved successfully.")

    def load_checkpoint(self, path: str):
        if not os.path.exists(path):
            logger.warning(f"⚠️ Checkpoint file not found: {path}")
            return False

        logger.info(f"📂 Loading brain state from {path}...")
        try:
            brain_state = torch.load(path, map_location=self.device)
            self.core_torch.load_state_dict(brain_state["model_state_dict"])
            self.astrocyte.current_energy = brain_state["astrocyte"].get("energy", 1000.0)
            self.pfc.current_goal = brain_state["pfc"].get("current_goal", "Rest")
            self.sleep_cycle_count = brain_state.get("sleep_cycle_count", 0)
            
            if "episodic_buffer" in brain_state:
                self.sleep_manager.episodic_buffer = brain_state["episodic_buffer"]
            
            if self.rag_system.vector_store_path:
                self.rag_system.load(self.rag_system.vector_store_path)

            logger.info("✅ Brain state loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load brain state: {e}")
            return False

    # --- Other Lifecycle Methods ---
    def process_tick(self, delta_time: float):
        if not self.is_awake: return
        self.astrocyte.consume_energy(1.0 * delta_time)
        if self.astrocyte.current_energy < 10.0:
            self.sleep()

    def sleep(self):
        logger.info(">>> 💤 Sleep Cycle Initiated <<<")
        self.is_awake = False
        self.sleep_cycle_count += 1
        self.astrocyte.replenish_energy(500.0)
        self.astrocyte.clear_fatigue(50.0)

    def wake_up(self):
        logger.info(">>> 🌅 Wake Up <<<")
        self.is_awake = True

    def perform_sleep_cycle(self, cycles: int = 1) -> Dict[str, int]:
        self.sleep()
        stats = self.sleep_manager.perform_maintenance(cycles)
        self.wake_up()
        return stats
    
    def run_cognitive_cycle(self, sensory_input: Any) -> Dict[str, Any]:
        return self.process_step(sensory_input)
    
    def get_brain_status(self) -> Dict[str, Any]:
        return {
            "state": "AWAKE" if self.is_awake else "SLEEPING",
            "cycle": self.sleep_cycle_count,
            "energy": self.astrocyte.current_energy,
            "fatigue": self.astrocyte.fatigue,
            "current_goal": self.pfc.current_goal
        }
    
    def reset_state(self):
        pass
    
    def retrieve_knowledge(self, query: str) -> List[str]:
        return self.rag_system.search(query)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core_torch(x)