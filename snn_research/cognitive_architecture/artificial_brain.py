# snn_research/cognitive_architecture/artificial_brain.py
# Title: Artificial Brain v21.21 (Simple Call)
# Description: 
#   SNNCore への呼び出しを単純な位置引数1つに戻す。
#   受け取り側が万能化したため、これで最も安全に動作する。

import torch
import torch.nn as nn
import logging
import os
from typing import Dict, Any, Optional, Union

from snn_research.core.snn_core import SpikingNeuralSubstrate
from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem
from snn_research.cognitive_architecture.rag_snn import RAGSystem
from snn_research.cognitive_architecture.prefrontal_cortex import PrefrontalCortex
from snn_research.cognitive_architecture.basal_ganglia import BasalGanglia
from snn_research.cognitive_architecture.motor_cortex import MotorCortex
from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator
from snn_research.cognitive_architecture.hippocampus import Hippocampus

logger = logging.getLogger(__name__)

class ArtificialBrain(nn.Module):
    def __init__(self, config: Dict[str, Any], device_name: Optional[str] = None, **kwargs):
        super().__init__()
        self.config = config
        target_device = device_name if device_name else config.get("device", "cpu")
        if target_device == "auto": target_device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.use_kernel = "event_driven" in str(config.get("training", {}).get("paradigm", ""))
        if self.use_kernel: target_device = "cpu"
        self.device = torch.device(target_device)
        
        self.is_awake = True
        self.sleep_cycle_count = 0
        self.d_model = config.get("model", {}).get("d_model", 128)
        
        self.core_torch = BitSpikeMamba(d_model=128, depth=2, vocab_size=128, expand=2, use_head=True).to(self.device)
        self.kernel_substrate = None
        if self.use_kernel:
            self.kernel_substrate = SpikingNeuralSubstrate(self.config, device="cpu")
            self.kernel_substrate.compile(self.core_torch)
            
        self.astrocyte = kwargs.get("astrocyte_network") or AstrocyteNetwork(max_energy=3000.0, device=str(self.device))
        self.workspace = kwargs.get("global_workspace") or GlobalWorkspace(dim=self.d_model, decay=0.9)
        self.motivation_system = IntrinsicMotivationSystem()
        self.rag_system = RAGSystem(embedding_dim=self.d_model, vector_store_path=os.path.join(config.get("runtime_dir", "./runtime_state"), "memory"))
        self.pfc = PrefrontalCortex(workspace=self.workspace, motivation_system=self.motivation_system, d_model=self.d_model, device=str(self.device))
        self.basal_ganglia = BasalGanglia(workspace=self.workspace)
        self.motor_cortex = MotorCortex(actuators=["default"], device=str(self.device))
        self.sleep_manager = SleepConsolidator(substrate=self.core_torch)
        self.hippocampus = Hippocampus(capacity=200, input_dim=self.d_model, device=str(self.device))
        
        logger.info(f"🚀 Artificial Brain v21.21 (Simple Call) initialized.")

    def wake_up(self):
        logger.info(">>> 🌅 Brain Wake Up sequence initiated <<<")
        self.is_awake = True
        self.astrocyte.replenish_energy(3000.0)
        if self.kernel_substrate: self.kernel_substrate.reset_state()
        return True

    def sleep(self):
        logger.info(">>> 💤 Brain Sleep sequence initiated <<<")
        self.is_awake = False
        self.sleep_cycle_count += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res_step = self.process_step(x)
        output = res_step.get("output")
        if not isinstance(output, torch.Tensor):
            output = torch.zeros(1, 128, device=self.device)
        return output

    def process_step(self, sensory_input: Any) -> Dict[str, Any]:
        if not self.is_awake or self.astrocyte.current_energy < 5.0:
            return {"status": "inactive", "output": torch.zeros(1, 128, device=self.device)}
        
        if torch.is_tensor(sensory_input): inp = sensory_input
        elif isinstance(sensory_input, dict) and "tensor" in sensory_input: inp = sensory_input["tensor"]
        else: inp = torch.randn(1, 128, device=self.device)

        uncertainty = 0.0
        
        if self.use_kernel and self.kernel_substrate:
            if inp.device.type != "cpu": inp = inp.cpu()
            
            # [SIMPLE CALL] 位置引数1つで呼ぶ。受け取り側が万能なのでこれでOK。
            output = self.kernel_substrate(inp)
            
            uncertainty = getattr(self.kernel_substrate, "uncertainty_score", 0.0)
            self.astrocyte.consume_energy(0.05 + (uncertainty * 1.0))
            output = output.to(self.device)
        else:
            eff_inp = inp.unsqueeze(1) if inp.dim() == 2 else inp
            output = self.core_torch(eff_inp)
            if isinstance(output, (list, tuple)): output = output[0]
            if output.dim() == 3: output = output[:, -1, :]
            self.astrocyte.consume_energy(0.5)
            
        self.workspace.step()
        return {
            "output": output, 
            "energy": self.astrocyte.current_energy, 
            "uncertainty": uncertainty,
            "status": "active"
        }

    def perform_sleep_cycle(self, cycles: int = 5):
        self.sleep()
        logger.info(f"💤 REM Sleep Phase (x{cycles})...")
        if self.use_kernel and self.kernel_substrate:
            self.kernel_substrate.kernel.is_sleeping = True
            for _ in range(cycles):
                self.kernel_substrate.forward_step({}, learning=True, dreaming=True)
            self.kernel_substrate.sleep_process()
            self.kernel_substrate.kernel.is_sleeping = False
        self.wake_up()

    @property
    def cycle_count(self): return self.sleep_cycle_count
    @property
    def state(self) -> str: return "AWAKE" if self.is_awake else "SLEEPING"
    def reset_state(self):
        if self.kernel_substrate: self.kernel_substrate.reset_state()