# ファイルパス: snn_research/cognitive_architecture/astrocyte_network.py
# 日本語タイトル: Astrocyte & Energy Management System
# 目的・内容:
#   神経活動によるエネルギー消費と、疲労の蓄積をシミュレートする。
#   エネルギー不足は活動抑制や強制睡眠（Shutdown）を引き起こす。
#   メタ認知スケジューラへの主要な入力となる。

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
        device: str = "cpu"
    ):
        super().__init__()
        self.max_energy = max_energy
        self.current_energy = max_energy
        self.fatigue = 0.0
        self.fatigue_threshold = fatigue_threshold
        
        # Parameters
        self.base_metabolism = 0.1
        self.spike_cost = 0.05
        self.recovery_rate = 2.0
    
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

    def step(self):
        """自然回復などの毎ステップ処理"""
        pass

    def get_diagnosis_report(self) -> Dict[str, Any]:
        """システムの健康状態レポート"""
        return {
            "status": "CRITICAL" if self.current_energy < 100 else "NORMAL",
            "metrics": {
                "energy": self.current_energy,
                "fatigue": self.fatigue,
                "max_energy": self.max_energy,
                "fatigue_threshold": self.fatigue_threshold
            }
        }