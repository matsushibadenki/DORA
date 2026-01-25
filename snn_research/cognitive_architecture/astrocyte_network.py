# ファイルパス: snn_research/cognitive_architecture/astrocyte_network.py
# 日本語タイトル: Astrocyte & Energy Management System (Fixed)
# 目的・内容:
#   神経活動によるエネルギー消費と、疲労の蓄積をシミュレートする。
#   Schedulerからの直接消費(consume_energy)に対応。

import torch
import torch.nn as nn
from typing import Dict, Any


class AstrocyteNetwork(nn.Module):
    """
    Manages global energy reservoir and metabolic waste (fatigue).
    """

    def __init__(
        self,
        max_energy: float = 1000.0,
        fatigue_threshold: float = 80.0,
        device: str = "cpu",
        **kwargs
    ):
        if 'initial_energy' in kwargs:
            max_energy = kwargs['initial_energy']
        super().__init__()
        self.max_energy = max_energy
        self.current_energy = max_energy
        self.fatigue = 0.0
        self.fatigue_threshold = fatigue_threshold

        # Parameters
        self.base_metabolism = 0.1
        self.spike_cost = 0.05
        self.recovery_rate = 2.0

        # Legacy / Test Compatibility
        self.modulators: Dict[str, float] = {}

    @property
    def energy(self) -> float:
        return self.current_energy

    @energy.setter
    def energy(self, value: float):
        self.current_energy = value

    def get_energy_level(self) -> float:
        """Return current energy level."""
        return self.current_energy

    def consume_energy(self, amount: float):
        """
        [OS API] 指定された量のエネルギーを消費し、疲労を蓄積する。
        Schedulerから呼び出される。
        """
        self.current_energy = max(0.0, self.current_energy - amount)
        self.fatigue += amount * 0.1  # 疲労も溜まる

    def request_resource(self, source: str, amount: float) -> bool:
        """Legacy method for resource request compatibility."""
        if self.current_energy > amount:
            self.current_energy -= amount
            return True
        return False

    def maintain_homeostasis(self, model: nn.Module, learning_rate: float):
        """Legacy method stub for test_homeostasis_scaling."""
        if self.modulators.get("glutamate", 0.0) > 0.8:
            with torch.no_grad():
                for param in model.parameters():
                    param.mul_(0.9)  # Scale down

    def handle_neuron_death(self, layer: nn.Module, death_rate: float):
        """Legacy method stub for test_neuron_death."""
        if hasattr(layer, 'weight'):
            with torch.no_grad():
                mask = torch.rand_like(
                    layer.weight) > death_rate  # type: ignore
                layer.weight.mul_(mask.float())  # type: ignore
            self.current_energy -= 10.0  # Repair cost

    def monitor_neural_activity(self, firing_rate: float):
        """
        神経発火頻度に基づいてエネルギーを消費し、疲労物質を蓄積する。
        """
        cost = self.base_metabolism + (firing_rate * self.spike_cost)
        self.current_energy -= cost
        self.current_energy = max(0.0, self.current_energy)

        # 疲労の蓄積 (活動が高いほど溜まる)
        waste_accumulation = cost * 0.5
        self.fatigue += waste_accumulation

    def replenish_energy(self, amount: float):
        """エネルギー補給（食事や睡眠による）"""
        self.current_energy += amount
        self.current_energy = min(self.max_energy, self.current_energy)

    def clear_fatigue(self, amount: float):
        """疲労物質の除去（睡眠による）"""
        self.fatigue -= amount
        self.fatigue = max(0.0, self.fatigue)

    def log_fatigue(self, amount: float):
        """疲労物質の直接蓄積（Guardrail等からの強制介入用）"""
        self.fatigue += amount

    def step(self):
        """自然回復などの毎ステップ処理"""
        pass

    def get_diagnosis_report(self) -> Dict[str, Any]:
        """システムの健康状態レポート"""
        return {
            "status": "CRITICAL" if self.current_energy < 100 else "NORMAL",
            "metrics": {
                "energy": self.current_energy,
                "current_energy": self.current_energy,  # For compatibility
                "fatigue": self.fatigue,
                "max_energy": self.max_energy,
                "fatigue_threshold": self.fatigue_threshold
            }
        }