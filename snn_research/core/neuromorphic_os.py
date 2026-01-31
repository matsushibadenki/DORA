# ファイルパス: snn_research/core/neuromorphic_os.py
# 日本語タイトル: Neuromorphic Research OS Kernel v8.2 (Refactored)
# 目的・内容:
#   OSカーネルのリファクタリング。
#   - HardwareAbstractionLayerの整理
#   - PyTorch Moduleとしての振る舞いを改善

import json
import logging
import os
import time
import asyncio
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

# --- Core Modules ---
from snn_research.core.snn_core import SpikingNeuralSubstrate
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
from snn_research.cognitive_architecture.neuromorphic_scheduler import NeuromorphicScheduler
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.hippocampus import Hippocampus
from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator

# --- Learning Rules ---
from snn_research.learning_rules.forward_forward import ForwardForwardRule
from snn_research.learning_rules.stdp import STDPRule
from snn_research.learning_rules.active_inference import ActiveInferenceRule

logger = logging.getLogger(__name__)


class HardwareAbstractionLayer:
    """
    ハードウェアリソース（CPU/GPU/MPS）の抽象化レイヤー。
    将来的にNeuromorphic Chipへのインターフェースもここで吸収する。
    """
    def __init__(self, request_device: Optional[str]):
        self.device = self._select_device(request_device)
        self.device_name = str(self.device)

    def _select_device(self, device_name: Union[str, None]) -> torch.device:
        if not device_name or device_name == "auto" or str(device_name).lower() == "none":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        try:
            return torch.device(device_name)
        except Exception as e:
            logger.warning(f"Hardware selection failed: {e}. Fallback to CPU.")
            return torch.device("cpu")


class NeuromorphicOS(nn.Module):
    """
    Neuromorphic Research OS (NROS) Kernel.
    脳型コンピューティングのための統合オペレーティングシステム。
    """

    def __init__(self, config: Dict[str, Any], device_name: Optional[str] = "auto"):
        super().__init__()
        self.config = config or {}

        # 1. Hardware
        self.hardware = HardwareAbstractionLayer(device_name)
        logger.info(
            f"🖥️  Neuromorphic OS booting on: {self.hardware.device_name}")

        # 2. Substrate (The Brain Tissue)
        self.substrate = SpikingNeuralSubstrate(
            self.config, self.hardware.device)
        self._build_research_substrate()

        # 3. Cognitive Modules
        self.global_workspace = GlobalWorkspace(
            dim=self.config.get("dim", 64)).to(self.hardware.device)

        self.hippocampus = Hippocampus(
            capacity=200,
            input_dim=self.config.get("input_dim", 784),
            device=str(self.hardware.device)
        )

        self.astrocyte = AstrocyteNetwork(
            max_energy=self.config.get("max_energy", 1000.0),
            device=str(self.hardware.device)
        ).to(self.hardware.device)

        # 4. System Managers
        self.scheduler = NeuromorphicScheduler(
            self.astrocyte, self.global_workspace)
        self.sleep_manager = SleepConsolidator(self.substrate)

        # 5. State Variables
        self.dopamine_level = 0.1
        self.base_dopamine = 0.1
        self.feedback_signal: Optional[torch.Tensor] = None
        self.system_status = "BOOTING"
        self.is_running = False
        self.cycle_count = 0

        # Observer
        self.state_dir = "workspace/runtime_state"
        os.makedirs(self.state_dir, exist_ok=True)
        self.state_file_path = os.path.join(
            self.state_dir, "brain_activity.json")

    @property
    def device(self) -> torch.device:
        return self.hardware.device

    def _build_research_substrate(self) -> None:
        """Forward-Forward, STDP, Active Inferenceのハイブリッド構成"""
        input_dim = self.config.get("input_dim", 784)
        hidden_dim = self.config.get("hidden_dim", 256)
        hippocampus_dim = self.config.get("hippocampus_dim", 128)
        output_dim = self.config.get("output_dim", 10)

        # Groups
        self.substrate.add_neuron_group("V1", input_dim)
        self.substrate.add_neuron_group("Association", hidden_dim)
        self.substrate.add_neuron_group("Hippocampus", hippocampus_dim)
        self.substrate.add_neuron_group("Motor", output_dim)

        # Rules
        # 各種学習則のインスタンス化
        # リファクタリングによりパラメータ名やインターフェースが整合していることを確認
        ff_rule = ForwardForwardRule(learning_rate=0.01, threshold=2.0)
        stdp_rule = STDPRule(learning_rate=0.05)
        active_inf = ActiveInferenceRule(learning_rate=0.005)

        # Projections
        self.substrate.add_projection(
            "v1_to_assoc", "V1", "Association", plasticity_rule=ff_rule)

        self.substrate.add_projection(
            "assoc_to_v1", "Association", "V1", plasticity_rule=active_inf)

        self.substrate.add_projection(
            "assoc_to_hippo", "Association", "Hippocampus", plasticity_rule=stdp_rule)
        self.substrate.add_projection(
            "hippo_to_assoc", "Hippocampus", "Association", plasticity_rule=stdp_rule)

        self.substrate.add_projection(
            "assoc_to_motor", "Association", "Motor", plasticity_rule=ff_rule)

        logger.info("🧠 Substrate configured: FF + STDP + Active Inference")

    def boot(self) -> None:
        self.substrate.reset_state()
        self.hippocampus.clear_memory()
        self.astrocyte.replenish_energy(1000.0)
        self.astrocyte.clear_fatigue(1000.0)
        self.cycle_count = 0
        self.system_status = "RUNNING"
        self.is_running = True
        logger.info("🚀 Neuromorphic OS Kernel started.")

    def reward(self, amount: float = 1.0) -> None:
        """外部報酬シグナルの注入"""
        self.dopamine_level += amount
        self.dopamine_level = min(self.dopamine_level, 5.0)

    def run_cycle(self, sensory_input: torch.Tensor, phase: str = "wake") -> Dict[str, Any]:
        """1認知サイクルの実行"""
        self.cycle_count += 1
        current_input = sensory_input.to(self.hardware.device)

        # 1. Biological Update
        self.astrocyte.step()
        self.dopamine_level = max(
            self.base_dopamine, self.dopamine_level * 0.95)

        substrate_inputs: Dict[str, torch.Tensor] = {}
        learning_phase = "neutral"

        # 2. Phase Logic
        if phase == "wake":
            substrate_inputs["V1"] = current_input
            if self.feedback_signal is not None:
                substrate_inputs["Association"] = self.feedback_signal * 0.5
            self.hippocampus.store_episode(current_input)
            
            # ドーパミンレベルが高い場合、学習フェーズをPositiveに（強化）
            if self.dopamine_level > 0.5:
                learning_phase = "positive"

        elif phase == "sleep":
            _ = self.sleep_manager.perform_maintenance(self.cycle_count)
            replay_signal = self.hippocampus.generate_replay(batch_size=1)

            if replay_signal is not None:
                substrate_inputs["V1"] = replay_signal.squeeze(0)
                learning_phase = "positive"
            else:
                noise = torch.randn_like(current_input) * 0.1
                substrate_inputs["V1"] = noise
                learning_phase = "negative"

            self.astrocyte.replenish_energy(10.0)
            self.astrocyte.clear_fatigue(5.0)
            self.feedback_signal = None

        # 3. Neural Computation
        target = current_input if phase == "wake" else substrate_inputs.get("V1")

        # forward_step は辞書を返す
        substrate_state = self.substrate.forward_step(
            substrate_inputs,
            phase=learning_phase,
            target_signal=target
        )

        # 発火統計の計算
        spikes_dict = substrate_state.get("spikes", {})
        total_spikes = sum(
            [s.sum().item() for s in spikes_dict.values() if s is not None]
        )
        self.astrocyte.monitor_neural_activity(firing_rate=total_spikes * 0.001)

        # 4. Cognitive Processing
        assoc_spikes = spikes_dict.get("Association")
        consciousness_level = 0.0

        if assoc_spikes is not None and phase == "wake":
            salience = assoc_spikes.mean().item() * 10.0 + self.dopamine_level
            self.global_workspace.upload_to_workspace(
                "Association", {"features": assoc_spikes}, salience=salience
            )

            thought = self.global_workspace.get_current_thought()
            consciousness_level = float(thought.mean().item())

            # 意識レベルが高い場合のみトップダウン信号を生成
            if consciousness_level > 0.01:
                # 次元チェック
                if thought.shape[-1] == assoc_spikes.shape[-1]:
                    self.feedback_signal = thought.detach()
            else:
                self.feedback_signal = None

        # 5. Scheduling & Logging
        scheduler_logs = self.scheduler.step()

        observation = self._pack_observation(
            phase, learning_phase, substrate_state, consciousness_level, scheduler_logs
        )
        self._export_state(observation)
        self._export_dashboard_data(observation)

        return observation

    def count_active_synapses(self) -> int:
        count = 0
        for p in self.substrate.parameters():
            if p.dim() > 1:
                count += int((p.abs() > 1e-6).sum().item())
        return count

    def _pack_observation(self, phase, learning_phase, state, conscious_lvl, logs) -> Dict[str, Any]:
        """観測データのパッケージング"""
        spikes = state.get("spikes", {})
        activity = {k: float(v.mean().item())
                    for k, v in spikes.items() if v is not None}
        bio = self.astrocyte.get_diagnosis_report()["metrics"]

        # Visual Data Export
        input_vis = []
        recon_vis = []

        if "V1" in spikes and spikes["V1"] is not None:
            v1_state = spikes["V1"]
            if v1_state.dim() > 1:
                v1_state = v1_state.mean(dim=0)
            input_vis = v1_state.tolist()

        if self.feedback_signal is not None:
            fb = self.feedback_signal
            if fb.dim() > 1:
                fb = fb.mean(dim=0)
            recon_vis = fb.tolist()
        elif "Association" in spikes and spikes["Association"] is not None:
            assoc = spikes["Association"]
            if assoc.dim() > 1:
                assoc = assoc.mean(dim=0)
            recon_vis = assoc.tolist()

        return {
            "timestamp": time.time(),
            "cycle": self.cycle_count,
            "phase": phase,
            "learning_phase": learning_phase,
            "bio_metrics": {**bio, "dopamine": self.dopamine_level},
            "substrate_activity": activity,
            "visual_cortex": {
                "input_image": input_vis,
                "reconstructed_image": recon_vis
            },
            "consciousness_level": conscious_lvl,
            "synapse_count": self.count_active_synapses(),
            "memory_stats": self.hippocampus.get_memory_stat(),
            "scheduler_log": [l["name"] for l in logs]
        }

    def _export_state(self, data: Dict[str, Any]) -> None:
        try:
            with open(self.state_file_path, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def _export_dashboard_data(self, data: Dict[str, Any]) -> None:
        dashboard_path = os.path.join(self.state_dir, "dashboard_data.json")
        try:
            with open(dashboard_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to export dashboard data: {e}")

    def shutdown(self) -> None:
        self.system_status = "SHUTDOWN"
        self._export_state({"status": "SHUTDOWN", "timestamp": time.time()})
        logger.info("💤 Neuromorphic OS shutting down.")

    # PyTorch Moduleとしての互換性
    def forward(self, x: torch.Tensor, phase: str = "wake", **kwargs: Any) -> Dict[str, Any]:
        return self.run_cycle(x, phase)

    async def sys_sleep(self, duration: float = 1.0) -> None:
        """
        System-wide sleep cycle trigger (Async).
        """
        logger.info(f"💤 SYS_SLEEP triggered for {duration}s...")
        self.system_status = "SLEEPING"

        cycles = int(duration * 10)
        # Dummy input for sleep cycle
        dim = int(self.config.get("input_dim", 784))
        
        for i in range(cycles):
            self.run_cycle(torch.zeros(dim), phase="sleep")
            await asyncio.sleep(0.1)

        self.system_status = "RUNNING"
        logger.info("☀️ SYS_SLEEP complete. Waking up.")