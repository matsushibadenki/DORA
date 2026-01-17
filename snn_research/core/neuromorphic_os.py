# ファイルパス: snn_research/core/neuromorphic_os.py
# 日本語タイトル: Neuromorphic Research OS Kernel v6.0 (Conscious Feedback)
# 目的・内容:
#   Neuromorphic OSの最上位コンテナ（最終形態）。
#   機能1: PyTorch標準機能を用いた確実なシナプス数カウント。
#   機能2: Global Workspaceからのトップダウン注意制御（Conscious Feedback）の実装。
#   意識が「見る」ことで、脳の活動を持続・強化させる閉ループを実現。

import json
import logging
import os
import time
import random
from typing import Any, Dict, List, Optional, Union
from collections import deque

import torch
import torch.nn as nn

# Core Layers
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
from snn_research.cognitive_architecture.neuromorphic_scheduler import NeuromorphicScheduler
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.core.snn_core import SpikingNeuralSubstrate

# Learning Rules
from snn_research.learning_rules.forward_forward import ForwardForwardRule
from snn_research.learning_rules.stdp import STDPRule

logger = logging.getLogger(__name__)


class HardwareAbstractionLayer:
    """[Layer 4] Neuromorphic Hardware Abstraction"""
    def __init__(self, request_device: Optional[str]):
        self.device = self._select_device(request_device)
        self.device_name = str(self.device)

    def _select_device(self, device_name: Union[str, None]) -> torch.device:
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
            logger.warning(f"Hardware selection failed: {e}. Fallback to CPU.")
            return torch.device("cpu")


class NeuromorphicOS(nn.Module):
    """Neuromorphic Research OS (NROS) v6.0"""

    def __init__(self, config: Dict[str, Any], device_name: Optional[str] = "auto"):
        super().__init__()
        self.config = config or {}

        # --- Hardware & Substrate ---
        self.hardware = HardwareAbstractionLayer(device_name)
        logger.info(f"🖥️ Neuromorphic OS booting on hardware: {self.hardware.device_name}")

        self.substrate = SpikingNeuralSubstrate(self.config, self.hardware.device)
        self._build_research_substrate()

        # --- Architecture ---
        self.global_workspace = GlobalWorkspace(dim=self.config.get("dim", 64))
        
        self.astrocyte = AstrocyteNetwork(
            max_energy=self.config.get("max_energy", 1000.0),
            fatigue_threshold=80.0,
            device=str(self.hardware.device)
        ).to(self.hardware.device)
        
        self.scheduler = NeuromorphicScheduler(self.astrocyte, self.global_workspace)

        # --- Memory & State ---
        self.hippocampal_buffer = deque(maxlen=200)
        self.last_substrate_state: Dict[str, Any] = {}
        self.dopamine_level = 0.0
        self.base_dopamine = 0.1
        
        # ★追加: 意識からのフィードバック信号（次サイクルの入力になる）
        self.feedback_signal: Optional[torch.Tensor] = None

        # Observer
        self.state_dir = "runtime_state"
        os.makedirs(self.state_dir, exist_ok=True)
        self.state_file_path = os.path.join(self.state_dir, "brain_activity.json")
        logger.info(f"📂 State observer linked to: {self.state_file_path}")

    @property
    def device(self) -> torch.device:
        return self.hardware.device

    def _build_research_substrate(self) -> None:
        input_dim = self.config.get("input_dim", 784)
        hidden_dim = self.config.get("hidden_dim", 256)
        hippocampus_dim = self.config.get("hippocampus_dim", 128)
        output_dim = self.config.get("output_dim", 10)

        self.substrate.add_neuron_group("V1", input_dim)
        self.substrate.add_neuron_group("Association", hidden_dim)
        self.substrate.add_neuron_group("Hippocampus", hippocampus_dim)
        self.substrate.add_neuron_group("Motor", output_dim)

        ff_rule = ForwardForwardRule(learning_rate=0.01, threshold=2.0)
        stdp_rule = STDPRule(learning_rate=0.05, tau_pre=20.0, tau_post=20.0)

        self.substrate.add_projection("v1_to_assoc", "V1", "Association", plasticity_rule=ff_rule)
        self.substrate.add_projection("assoc_to_hippo", "Association", "Hippocampus", plasticity_rule=stdp_rule)
        self.substrate.add_projection("hippo_to_assoc", "Hippocampus", "Association", plasticity_rule=stdp_rule)
        self.substrate.add_projection("assoc_to_motor", "Association", "Motor", plasticity_rule=ff_rule)

        logger.info("🧠 Substrate configured: Hybrid Learning (FF + STDP)")

    def boot(self) -> None:
        """システム起動"""
        self.substrate.reset_state()
        self.scheduler.clear_queue()
        self.astrocyte.replenish_energy(1000.0)
        self.astrocyte.clear_fatigue(1000.0)
        self.hippocampal_buffer.clear()
        self.last_substrate_state = {}
        self.dopamine_level = self.base_dopamine
        self.feedback_signal = None
        
        self.system_status = "RUNNING"
        self.cycle_count = 0
        logger.info("🚀 Neuromorphic OS Kernel started.")

    def reward(self, amount: float = 1.0):
        self.dopamine_level += amount
        self.dopamine_level = min(self.dopamine_level, 5.0)

    def count_active_synapses(self) -> int:
        """
        [修正版 v2] PyTorchの全パラメータ走査による確実なカウント
        """
        total = 0
        # モデル内の全パラメータ（重み）を走査
        for name, param in self.substrate.named_parameters():
            if "weight" in name and param.dim() > 1: # 重み行列のみ対象（バイアス除外）
                # 閾値以上の結合をカウント
                count = (torch.abs(param) > 1e-4).sum().item()
                total += count
        return total

    def sleep_pruning(self):
        """[Sleep Feature] Synaptic Pruning"""
        # 全パラメータに対して刈り込み適用
        for name, param in self.substrate.named_parameters():
            if "weight" in name and param.dim() > 1:
                threshold = 0.05
                mask = torch.abs(param.data) > threshold
                param.data *= mask.float()

    def synaptogenesis(self):
        """[Sleep Feature] Synaptogenesis"""
        for name, param in self.substrate.named_parameters():
            if "weight" in name and param.dim() > 1:
                zero_mask = (torch.abs(param.data) < 1e-4)
                birth_rate = 0.01
                birth_mask = (torch.rand_like(param.data) < birth_rate) & zero_mask
                new_connections = torch.randn_like(param.data) * 0.1
                param.data += new_connections * birth_mask.float()

    def run_cycle(self, sensory_input: torch.Tensor, phase: str = "wake") -> Dict[str, Any]:
        """
        1タイムステップを実行。
        v6.0: 意識からのトップダウン信号を入力に追加。
        """
        self.cycle_count += 1
        current_input = sensory_input.to(self.hardware.device)

        # 1. Phase & Homeostasis
        learning_phase = "neutral"
        self.astrocyte.step()
        self.dopamine_level = max(self.base_dopamine, self.dopamine_level * 0.95)

        # 入力データの構築
        substrate_inputs = {}

        if phase == "wake":
            # ボトムアップ入力 (V1)
            substrate_inputs["V1"] = current_input
            
            # ★トップダウン入力 (意識からのフィードバック)
            # 前回のサイクルで意識が生成した信号を、連合野(Association)へのバイアスとして注入
            if self.feedback_signal is not None:
                # 次元調整が必要な場合はリサイズ（簡易実装）
                # ここではAssociation野への直接入力として扱う
                substrate_inputs["Association"] = self.feedback_signal * 0.5 # 強度調整

            if self.dopamine_level > 0.5:
                learning_phase = "positive"
            
            if current_input.sum() > 0.1:
                self.hippocampal_buffer.append(current_input.detach().cpu())

        elif phase == "sleep":
            # 睡眠時: 構造可塑性 + リプレイ
            if self.cycle_count % 10 == 0:
                self.sleep_pruning()
                self.synaptogenesis()

            if len(self.hippocampal_buffer) > 10:
                memory_trace = random.choice(self.hippocampal_buffer).to(self.hardware.device)
                noise = torch.randn_like(memory_trace) * 0.05
                substrate_inputs["V1"] = memory_trace + noise
                learning_phase = "positive"
            else:
                noise = torch.randn_like(current_input) * 0.1
                substrate_inputs["V1"] = noise
                learning_phase = "negative"

            self.astrocyte.replenish_energy(12.0)
            self.astrocyte.clear_fatigue(6.0)
            self.feedback_signal = None # 睡眠中は意識フィードバックをリセット

        # 2. SNN Execution
        substrate_state = self.substrate.forward_step(
            substrate_inputs, 
            phase=learning_phase
        )
        self.last_substrate_state = substrate_state

        # Energy Cost
        total_spikes = sum([s.sum().item() for s in substrate_state["spikes"].values() if s is not None])
        self.astrocyte.monitor_neural_activity(firing_rate=total_spikes * 0.001)

        # 3. Conscious Processing (Global Workspace)
        assoc_spikes = substrate_state["spikes"].get("Association")
        consciousness_level = 0.0
        
        if assoc_spikes is not None and phase == "wake":
            # (A) Broadcast: 連合野 -> 意識
            salience = assoc_spikes.float().mean().item() * 10.0 + (self.dopamine_level * 0.5)
            self.global_workspace.upload_to_workspace(
                "Association", {"features": assoc_spikes}, salience=salience
            )
            
            # (B) Feedback Generation: 意識 -> 連合野 (Top-Down Attention)
            # 現在の意識状態を取得
            current_thought = self.global_workspace.get_current_thought()
            consciousness_level = float(current_thought.mean().item())
            
            # 意識レベルが高い場合、その思考パターンをフィードバック信号として保存
            if consciousness_level > 0.01:
                # 思考ベクトルを連合野の入力サイズに合わせて変形・投影する処理が必要だが
                # ここでは簡易的に直結（次元が合う前提、またはブロードキャスト）
                # 次元が合わない場合はゼロ埋め等で対応するロジックが必要だが、SNN側で吸収させる
                if current_thought.shape[-1] == assoc_spikes.shape[-1]:
                     self.feedback_signal = current_thought.detach()
                else:
                    # 次元不一致ならフィードバックなし（安全策）
                    self.feedback_signal = None
            else:
                self.feedback_signal = None

        scheduler_results = self.scheduler.step()

        # 4. Observation
        activity_summary = {
            k: float(v.float().mean().item()) 
            for k, v in substrate_state["spikes"].items() 
            if v is not None
        }
        
        bio_report = self.astrocyte.get_diagnosis_report()
        synapse_count = self.count_active_synapses() # 修正版カウンタ

        observation = {
            "timestamp": time.time(),
            "cycle": self.cycle_count,
            "status": self.system_status,
            "phase": phase,
            "learning_phase": learning_phase,
            "bio_metrics": {
                **bio_report["metrics"],
                "dopamine": self.dopamine_level
            },
            "substrate_activity": activity_summary,
            "consciousness_level": consciousness_level,
            "synapse_count": synapse_count,
            "scheduler_log": [r["name"] for r in scheduler_results]
        }

        self._export_state(observation)
        return observation

    def _export_state(self, observation: Dict[str, Any]):
        try:
            with open(self.state_file_path, "w") as f:
                json.dump(observation, f)
        except Exception:
            pass

    def shutdown(self) -> None:
        self.system_status = "SHUTDOWN"
        self._export_state({"status": "SHUTDOWN", "timestamp": time.time()})
        logger.info("💤 Neuromorphic OS shutting down.")