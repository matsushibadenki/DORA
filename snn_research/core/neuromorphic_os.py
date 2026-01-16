# ファイルパス: snn_research/core/neuromorphic_os.py
# 修正: _select_device メソッドの堅牢化

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, List, Union

from snn_research.core.snn_core import SpikingNeuralSubstrate
# 仮のインポート（必要に応じて実体に合わせてください）
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace

logger = logging.getLogger(__name__)

class NeuromorphicOS(nn.Module):
    """
    Neuromorphic Research OS v1.0
    """
    
    def __init__(self, config: Dict[str, Any], device_name: Optional[str] = "auto"):
        super().__init__()
        self.config = config or {} # configがNoneの場合に備える
        
        # 1. Hardware Initialization
        self.device = self._select_device(device_name)
        logger.info(f"🖥️ Neuromorphic OS booting on device: {self.device}")
        
        # 2. Kernel (Substrate) Initialization
        self.kernel = SpikingNeuralSubstrate(self.config, self.device)
        self._build_default_substrate()
        
        # 3. Cognitive Modules
        self.global_workspace = GlobalWorkspace()
        
        # 4. Observer
        self.system_status = "BOOTING"
        self.cycle_count = 0

    def _select_device(self, device_name: Union[str, None]) -> torch.device:
        """デバイス選択ロジック (None安全)"""
        # None, "None", 空文字, "auto" の場合は自動選択
        if not device_name or device_name == "auto" or str(device_name).lower() == "none":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        
        try:
            return torch.device(device_name)
        except RuntimeError as e:
            logger.warning(f"⚠️ Invalid device '{device_name}' specified. Falling back to CPU. Error: {e}")
            return torch.device("cpu")

    def _build_default_substrate(self):
        """初期構成のビルド"""
        input_dim = self.config.get("input_dim", 784)
        hidden_dim = self.config.get("hidden_dim", 256)
        output_dim = self.config.get("output_dim", 10)
        
        self.kernel.add_neuron_group("V1", input_dim)
        self.kernel.add_neuron_group("Association", hidden_dim)
        self.kernel.add_neuron_group("Motor", output_dim)
        
        self.kernel.add_projection("ff_v1_assoc", "V1", "Association")
        self.kernel.add_projection("ff_assoc_motor", "Association", "Motor")
        
        logger.info("🧠 Default substrate topology built.")

    def boot(self):
        """システム起動シーケンス"""
        self.kernel.reset_state()
        self.system_status = "RUNNING"
        logger.info("🚀 Neuromorphic OS Kernel started.")

    def run_cycle(self, sensory_input: torch.Tensor) -> Dict[str, Any]:
        """OSメインループ"""
        self.cycle_count += 1
        inputs = {"V1": sensory_input.to(self.device)}
        substrate_state = self.kernel.forward_step(inputs)
        return {
            "cycle": self.cycle_count,
            "status": self.system_status,
            "substrate_state": substrate_state,
            "action": None
        }

    def shutdown(self):
        self.system_status = "SHUTDOWN"
        logger.info("💤 Neuromorphic OS shutting down.")