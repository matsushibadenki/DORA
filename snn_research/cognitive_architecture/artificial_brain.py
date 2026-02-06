# snn_research/cognitive_architecture/artificial_brain.py
# Title: Artificial Brain v18.8 (Recursion Fix)
# Description: self.cortex = self による無限再帰(RecursionError)を修正。

import torch
import torch.nn as nn
import logging
import random
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

# Optional
try:
    from snn_research.cognitive_architecture.theory_of_mind import TheoryOfMind
    from snn_research.cognitive_architecture.causal_inference_engine import CausalInferenceEngine
    ADVANCED_MODULES_AVAILABLE = True
except ImportError:
    ADVANCED_MODULES_AVAILABLE = False

logger = logging.getLogger(__name__)

class ArtificialBrain(nn.Module):
    def __init__(self, config: Dict[str, Any], device_name: Optional[str] = None):
        super().__init__()
        self.config = config

        target_device = device_name
        if not target_device or target_device == "auto" or str(target_device) == "None":
            target_device = config.get("device")
        
        training_conf = config.get("training", {})
        if hasattr(training_conf, "paradigm"):
            paradigm = str(training_conf.paradigm)
        else:
            paradigm = str(training_conf.get("paradigm", "gradient_based"))
            
        self.use_kernel = "event_driven" in paradigm

        if self.use_kernel:
            target_device = "cpu"
            print(f"⚡ [Brain] Kernel Mode Detected. Using simplified MLP structure.")
        elif not target_device or target_device == "auto":
            target_device = "cpu"
        
        self.device = torch.device(target_device)
        self.is_awake = True
        self.plasticity_enabled = False
        self.sleep_cycle_count = 0
        self.d_model = config.get("model", {}).get("d_model", 256)
        
        self.boredom_counter = 0.0
        self.boredom_threshold = 10.0

        self._init_core_brain_torch()
        
        self.kernel_substrate: Optional[SpikingNeuralSubstrate] = None
        
        if self.use_kernel:
            print("⚡ [Brain] Switching to Event-Driven Mode (No-Matrix/No-BP)...")
            self._compile_to_kernel()
        else:
            logger.info(f"ℹ️  Running in Standard PyTorch Mode (Matrix Ops). Paradigm: {paradigm}")

        self.astrocyte = AstrocyteNetwork(max_energy=1000.0, fatigue_threshold=80.0, device=str(self.device))

        self.workspace = GlobalWorkspace(dim=self.d_model, decay=config.get("workspace_decay", 0.9))
        self.motivation_system = IntrinsicMotivationSystem(curiosity_weight=config.get("curiosity_weight", 1.0))
        self.rag_system = RAGSystem(embedding_dim=self.d_model, vector_store_path=config.get("memory_path", "./runtime_state/memory"))
        self.hippocampus = Hippocampus(capacity=200, input_dim=self.d_model, device=str(self.device))
        self.pfc = PrefrontalCortex(workspace=self.workspace, motivation_system=self.motivation_system, d_model=self.d_model, device=str(self.device))
        self.basal_ganglia = BasalGanglia(workspace=self.workspace, selection_threshold=0.3)
        self.motor_cortex = MotorCortex(actuators=["default_actuator"], device=str(self.device))

        self.theory_of_mind = None
        self.causal_engine = None
        if ADVANCED_MODULES_AVAILABLE and config.get("enable_advanced_cognition", True):
            self.theory_of_mind = TheoryOfMind(self.workspace, self.rag_system)
            self.causal_engine = CausalInferenceEngine(self.rag_system, self.workspace)

        self.sleep_manager = SleepConsolidator(substrate=self.core_torch)
        
        # Legacy Attribute alias
        self.thinking_engine = self.core_torch
        # [Fix] self.cortex = self を削除し、プロパティで実装する (再帰エラー防止)

        logger.info(f"🚀 Artificial Brain v18.8 initialized on {self.device}.")

    @property
    def cortex(self):
        """Legacy alias for self"""
        return self

    def _init_core_brain_torch(self):
        model_conf = self.config.get("model", {})
        arch_type = model_conf.get("architecture_type", "sformer")

        if self.use_kernel:
            input_dim = 128
            hidden_dim = 512
            output_dim = self.d_model
            
            self.core_torch = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
            
            for m in self.core_torch.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=2.0)
                    if m.bias is not None: nn.init.constant_(m.bias, 0.0)

            print(f"⚡ [Brain] Kernel Model defined: Linear({input_dim}->{hidden_dim}) -> Linear({hidden_dim}->{output_dim})")
            
        elif arch_type == "bio_pc_network":
            layer_sizes = model_conf.get("network", {}).get("layer_sizes", [784, 512, 256, self.d_model])
            self.core_torch = BioPCNetwork(input_shape=(784,), layer_sizes=layer_sizes, config=self.config)
        elif arch_type == "sformer":
            self.core_torch = ScaleAndFireTransformer(
                vocab_size=self.config.get("data", {}).get("vocab_size", 50257),
                d_model=self.d_model,
                num_layers=model_conf.get("num_layers", 4),
            )
        else:
            self.core_torch = BioPCNetwork(input_shape=(784,), layer_sizes=[64, 128, self.d_model], config=self.config)

        self.core_torch.to(self.device)

    def _compile_to_kernel(self):
        try:
            print("🔨 [Compiler] Building DORA Kernel Graph from PyTorch model...")
            self.kernel_substrate = SpikingNeuralSubstrate(self.config, device="cpu") 
            cpu_model = self.core_torch.to("cpu")
            self.kernel_substrate.kernel.build_from_torch_model(cpu_model)
            self.kernel_substrate.kernel_compiled = True
            self.core_torch.to(self.device)
            
            total_neurons = len(self.kernel_substrate.kernel.neurons)
            if total_neurons > 0:
                input_size = 128 
                output_size = self.d_model
                self.kernel_substrate.group_indices["visual_cortex"] = (0, min(input_size, total_neurons))
                self.kernel_substrate.group_indices["output_cortex"] = (max(0, total_neurons - output_size), total_neurons)
                print(f"✅ [Compiler] SUCCESS: {total_neurons} neurons created. {self.kernel_substrate.kernel.stats['ops']} synapses ready.")
            else:
                print("❌ [Compiler] CRITICAL: 0 neurons created! Model has no visible nn.Linear layers.")
                self.use_kernel = False
        except Exception as e:
            logger.error(f"❌ Kernel Compilation Failed: {e}")
            import traceback
            traceback.print_exc()
            self.use_kernel = False

    @property
    def cycle_count(self) -> int:
        return self.sleep_cycle_count

    @property
    def device_name(self) -> str:
        return str(self.device)
        
    @property
    def state(self) -> str:
        return "AWAKE" if self.is_awake else "SLEEPING"

    def process_tick(self, delta_time: float):
        if not self.is_awake: return
        self.astrocyte.consume_energy(1.0 * delta_time)
        self.boredom_counter += delta_time
        
        if self.astrocyte.current_energy < 10.0:
            logger.info("⚠️ Critical Energy Low. Auto-triggering sleep cycle.")
            self.sleep()
            return

        if self.boredom_counter > self.boredom_threshold:
            self._trigger_spontaneous_thought()
            self.boredom_counter = 0.0

    def _trigger_spontaneous_thought(self):
        pass
    
    def reset_state(self):
        if self.use_kernel and self.kernel_substrate:
            self.kernel_substrate.reset_state()
        if hasattr(self.core_torch, "reset_state"):
             if callable(self.core_torch.reset_state):
                 self.core_torch.reset_state()

    def get_brain_status(self) -> Dict[str, Any]:
        current_thought = self.workspace.get_current_thought()
        thought_str = "None"
        if current_thought is not None:
            thought_str = "Active"

        return {
            "state": self.state,
            "cycle": self.sleep_cycle_count,
            "energy": self.astrocyte.current_energy,
            "fatigue": self.astrocyte.fatigue,
            "current_goal": self.pfc.current_goal,
            "drives": self.motivation_system.get_internal_state(),
            "conscious_content": thought_str,
            "mode": "Event-Driven" if self.use_kernel else "Matrix-Based"
        }

    def process_step(self, sensory_input: Union[torch.Tensor, str, Dict[str, Any]]) -> Dict[str, Any]:
        self.boredom_counter = 0.0
        
        if not self.is_awake:
            if self.use_kernel and self.kernel_substrate:
                if hasattr(self.kernel_substrate.kernel, "apply_synaptic_scaling"):
                    self.kernel_substrate.kernel.apply_synaptic_scaling(0.99)
                
                if isinstance(sensory_input, torch.Tensor):
                    inputs = {"visual_cortex": sensory_input}
                    self.kernel_substrate.forward_step(inputs)
                    
                return {"status": "dreaming", "energy": self.astrocyte.current_energy}
            else:
                return {"status": "sleeping"}

        if self.astrocyte.current_energy <= 5.0:
            return {"status": "fatigued", "output": None}

        output_tensor = None
        perception_content = {}
        
        if self.use_kernel and self.kernel_substrate:
            if isinstance(sensory_input, torch.Tensor):
                inputs = {"visual_cortex": sensory_input}
                kernel_result = self.kernel_substrate.forward_step(inputs)
                spikes = kernel_result["spikes"]
                if "output_cortex" in spikes:
                    output_tensor = spikes["output_cortex"]
                else:
                    output_tensor = list(spikes.values())[-1] if spikes else torch.zeros(1, self.d_model)
                perception_content = {"features": output_tensor, "modality": "visual"}
        else:
            if isinstance(sensory_input, torch.Tensor):
                sensory_input = sensory_input.to(self.device)
                with torch.no_grad():
                    try:
                        if hasattr(self.core_torch, 'input_shape') or isinstance(self.core_torch, nn.Sequential):
                             if sensory_input.shape[-1] != 128:
                                 pass
                        model_output = self.core_torch(sensory_input)
                        if isinstance(model_output, tuple): features = model_output[0]
                        elif isinstance(model_output, dict): features = next((v for v in model_output.values() if isinstance(v, torch.Tensor)), torch.zeros_like(sensory_input))
                        else: features = model_output
                        output_tensor = features
                    except Exception as e:
                        logger.error(f"Model Forward Failed: {e}")
                        output_tensor = torch.zeros(1, self.d_model, device=self.device)
                perception_content = {"features": output_tensor, "modality": "visual"}
        
        if self.is_awake and output_tensor is not None:
            self.workspace.upload_to_workspace(source_name="sensory_cortex", content=perception_content, salience=0.8)

        self.workspace.step()
        conscious_content = self.workspace.get_current_content()
        
        surprise = 0.1
        if output_tensor is not None and output_tensor.numel() > 0:
            surprise = output_tensor.std().item()
            
        current_drives = self.motivation_system.process(sensory_input, prediction_error=surprise)
        
        pfc_plan = self.pfc.plan(conscious_content)
        action_candidates = []
        if pfc_plan:
            action_candidates.append({"action": pfc_plan.get("directive", "wait"), "value": pfc_plan.get("priority", 0.5)})

        selected_action: Union[str, Dict[str, Any], None] = "rest"
        
        if self.is_awake:
            selected_action = self.basal_ganglia.select_action(external_candidates=action_candidates, emotion_context=current_drives)
            self.astrocyte.consume_energy(2.0)

        return {
            "output": output_tensor,
            "conscious_broadcast": conscious_content,
            "pfc_goal": self.pfc.current_goal,
            "drives": current_drives,
            "action": selected_action,
            "energy": self.astrocyte.current_energy,
            "executed_modules": ["Core", "GlobalWorkspace", "PFC", "BG"]
        }

    def run_cognitive_cycle(self, sensory_input: Any) -> Dict[str, Any]:
        return self.process_step(sensory_input)

    def sleep_cycle(self):
        self.sleep()

    def retrieve_knowledge(self, query: str) -> List[str]:
        if hasattr(self.rag_system, 'search'): return self.rag_system.search(query)
        return []

    def sleep(self): 
        logger.info(">>> 💤 Sleep Cycle Initiated (Consolidating & Pruning) <<<")
        self.is_awake = False
        self.sleep_cycle_count += 1
        
        if self.use_kernel and self.kernel_substrate:
            self.kernel_substrate.kernel.set_sleep_mode(True)
            self.kernel_substrate.reset_state()
            
        self.astrocyte.replenish_energy(500.0)
        self.astrocyte.clear_fatigue(50.0)
        if self.sleep_manager: self.sleep_manager.perform_maintenance(self.sleep_cycle_count)

    def wake_up(self): 
        logger.info(">>> 🌅 Wake Up (Synaptogenesis Enabled) <<<")
        self.is_awake = True
        
        if self.use_kernel and self.kernel_substrate:
            self.kernel_substrate.kernel.set_sleep_mode(False)
            
    def set_plasticity(self, active: bool): self.plasticity_enabled = active
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.process_step(x)
        out = res.get("output")
        if out is None: return torch.zeros(1, self.d_model, device=self.device)
        return out