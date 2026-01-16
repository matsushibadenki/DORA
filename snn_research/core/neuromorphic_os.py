# ファイルパス: snn_research/core/neuromorphic_os.py
# 日本語タイトル: Neuromorphic Research OS Kernel v4.1 (State Export Enabled)
# 目的・内容:
#   Neuromorphic OSの最上位コンテナ。
#   修正: ダッシュボード連携のため、run_cycle毎に内部状態をJSONファイルへエクスポートする機能を追加。
#   これにより、別プロセスのObserverがリアルタイムに状態を読み取れるようになる。

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

# Core Layers
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
from snn_research.cognitive_architecture.neuromorphic_scheduler import NeuromorphicScheduler
from snn_research.core.snn_core import SpikingNeuralSubstrate

# Learning Rules
from snn_research.learning_rules.forward_forward import ForwardForwardRule
from snn_research.learning_rules.stdp import STDPRule

logger = logging.getLogger(__name__)


class HardwareAbstractionLayer:
    """
    ハードウェア依存部分を吸収するレイヤ。
    CPU/GPU/MPS(Mac) などを自動判定し、デバイスオブジェクトを提供する。
    """

    def __init__(self, request_device: Optional[str]):
        self.device = self._select_device(request_device)
        self.device_name = str(self.device)

    def _select_device(self, device_name: Union[str, None]) -> torch.device:
        """デバイス選択ロジック"""
        if not device_name or device_name == "auto" or str(device_name).lower() == "none":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        try:
            return torch.device(device_name)
        except Exception as e:
            logger.warning(f"Device selection failed: {e}. Fallback to CPU.")
            return torch.device("cpu")


class NeuromorphicOS(nn.Module):
    """
    Neuromorphic Research OS (NROS) v4.1
    
    知能現象を「実装」するのではなく、神経ダイナミクスとして「観測」するための統合基盤。
    Forward-Forward則とSTDP則が同一基盤上で共存する異種混合アーキテクチャを提供する。
    """

    def __init__(self, config: Dict[str, Any], device_name: Optional[str] = "auto"):
        super().__init__()
        self.config = config or {}

        # --- Layer 1: Hardware Abstraction ---
        self.hardware = HardwareAbstractionLayer(device_name)
        logger.info(f"🖥️ Neuromorphic OS booting on hardware: {self.hardware.device_name}")

        # --- Layer 2: Spiking Neural Substrate (The Kernel) ---
        self.substrate = SpikingNeuralSubstrate(self.config, self.hardware.device)
        self._build_heterogeneous_substrate()

        # --- Layer 3: Cognitive Architecture ---
        self.global_workspace = GlobalWorkspace(dim=self.config.get("dim", 64))
        
        # Scheduler用のアストロサイト（簡易エネルギーモデル）
        class AstrocyteSimulator:
            def __init__(self, max_energy=100.0):
                self.max_energy = max_energy
                self.current_energy = max_energy
                self.fatigue_toxin = 0.0 # 疲労毒素（アデノシンなど）

            def request_resource(self, name: str, amount: float) -> bool:
                if self.current_energy >= amount:
                    self.current_energy -= amount
                    self.fatigue_toxin += amount * 0.1
                    return True
                return False

            def recover(self):
                """睡眠時の回復プロセス"""
                self.current_energy = min(self.max_energy, self.current_energy + 10.0)
                self.fatigue_toxin = max(0.0, self.fatigue_toxin - 5.0)

        self.astrocyte = AstrocyteSimulator(max_energy=300.0)
        self.scheduler = NeuromorphicScheduler(self.astrocyte, self.global_workspace)

        # --- Layer 4: Observation & Logging (System State Export) ---
        self.system_status = "BOOTING"
        self.cycle_count = 0
        
        # 状態共有用のディレクトリ作成
        self.state_dir = "runtime_state"
        os.makedirs(self.state_dir, exist_ok=True)
        self.state_file_path = os.path.join(self.state_dir, "brain_activity.json")
        logger.info(f"📂 State observer linked to: {self.state_file_path}")

    @property
    def device(self) -> torch.device:
        """互換性のためのプロパティ"""
        return self.hardware.device

    def _build_heterogeneous_substrate(self) -> None:
        """
        異種混合ネットワークの構築
        - Cortex (V1, Association): Forward-Forward Learning
        - Hippocampus: STDP (Temporal Association)
        - Motor: Forward-Forward
        """
        input_dim = self.config.get("input_dim", 784)
        hidden_dim = self.config.get("hidden_dim", 256)
        hippocampus_dim = self.config.get("hippocampus_dim", 128)
        output_dim = self.config.get("output_dim", 10)

        # 1. 領域の作成
        self.substrate.add_neuron_group("V1", input_dim)
        self.substrate.add_neuron_group("Association", hidden_dim)
        self.substrate.add_neuron_group("Hippocampus", hippocampus_dim)
        self.substrate.add_neuron_group("Motor", output_dim)

        # 2. 学習則のインスタンス化
        ff_rule = ForwardForwardRule(learning_rate=0.01, threshold=2.0)
        stdp_rule = STDPRule(learning_rate=0.05, tau_pre=20.0, tau_post=20.0)

        # 3. 投射（Connectome）の作成
        self.substrate.add_projection("v1_to_assoc", "V1", "Association", plasticity_rule=ff_rule)
        self.substrate.add_projection("assoc_to_hippo", "Association", "Hippocampus", plasticity_rule=stdp_rule)
        self.substrate.add_projection("hippo_to_assoc", "Hippocampus", "Association", plasticity_rule=stdp_rule)
        self.substrate.add_projection("assoc_to_motor", "Association", "Motor", plasticity_rule=ff_rule)

        logger.info("🧠 Heterogeneous neural substrate built: Cortex(FF) + Hippocampus(STDP)")

    def boot(self) -> None:
        """システム起動シーケンス"""
        self.substrate.reset_state()
        self.system_status = "RUNNING"
        self.cycle_count = 0
        logger.info("🚀 Neuromorphic OS Kernel started. Ready for experiments.")

    def run_cycle(self, sensory_input: torch.Tensor, phase: str = "wake") -> Dict[str, Any]:
        """
        OSのメインループ（1タイムステップ）。
        状態を更新し、その結果をJSONファイルにも書き出す。
        """
        self.cycle_count += 1
        current_input = sensory_input.to(self.hardware.device)

        # 1. Input Processing & Phase Control
        substrate_inputs = {}
        
        if phase == "wake":
            substrate_inputs["V1"] = current_input
            learning_phase = "positive"
            
        elif phase == "sleep":
            # 睡眠時: 夢（ノイズ）の入力と回復
            noise = torch.randn_like(current_input) * 0.1
            substrate_inputs["V1"] = noise
            self.astrocyte.recover()
            learning_phase = "negative"
        else:
            learning_phase = "neutral"

        # 2. Substrate Step
        substrate_state = self.substrate.forward_step(
            substrate_inputs, 
            phase=learning_phase
        )

        # 3. Cognitive Services
        assoc_spikes = substrate_state["spikes"].get("Association")
        if assoc_spikes is not None:
            self.global_workspace.upload_to_workspace(
                "Association", {"features": assoc_spikes}, salience=0.8
            )

        # 4. Observation & Serialization
        # テンソルをPythonの数値型に変換して辞書を作成
        activity_summary = {
            k: float(v.float().mean().item()) 
            for k, v in substrate_state["spikes"].items() 
            if v is not None
        }
        
        consciousness_level = float(self.global_workspace.get_current_thought().mean().item())

        observation = {
            "timestamp": time.time(),
            "cycle": self.cycle_count,
            "status": self.system_status,
            "phase": phase,
            "learning_phase": learning_phase,
            "energy": self.astrocyte.current_energy,
            "fatigue": self.astrocyte.fatigue_toxin,
            "substrate_activity": activity_summary,
            "consciousness": consciousness_level,
        }

        # 状態をファイルへ書き出し (Observer用)
        try:
            with open(self.state_file_path, "w") as f:
                json.dump(observation, f)
        except Exception as e:
            logger.warning(f"Failed to export brain state: {e}")

        return observation

    def shutdown(self) -> None:
        """システム終了処理"""
        self.system_status = "SHUTDOWN"
        # シャットダウン状態も書き出しておく
        try:
            with open(self.state_file_path, "w") as f:
                json.dump({"status": "SHUTDOWN", "timestamp": time.time()}, f)
        except Exception:
            pass
        logger.info("💤 Neuromorphic OS shutting down.")