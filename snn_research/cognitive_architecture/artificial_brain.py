# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 変更点:
# - AstrocyteNetwork の統合 (エネルギー管理)
# - SleepConsolidator の統合 (睡眠時の構造改革・剪定)
# - get_brain_status でリアルな値を返すように修正

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, cast, Union

# Phase 2 Components
from snn_research.core.networks.bio_pc_network import BioPCNetwork
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator

# Legacy/Alternative models
from snn_research.models.transformer.sformer import ScaleAndFireTransformer 

logger = logging.getLogger(__name__)

class ArtificialBrain(nn.Module):
    """
    Artificial Brain v14.0 (Phase 2 Integrated)
    
    Responsibilities:
    - Hosts the Core SNN (Bio-PCNet)
    - Manages Energy (Astrocyte)
    - Manages Structure (Sleep Consolidation)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))
        
        self.is_awake = True
        self.plasticity_enabled = False
        self.sleep_cycle_count = 0
        
        # アーキテクチャの初期化
        self._init_core_brain()
        
        # 1. Astrocyte Network (Energy Management)
        self.astrocyte = AstrocyteNetwork(
            max_energy=1000.0,
            fatigue_threshold=80.0,
            device=str(self.device)
        )
        
        # 2. Sleep Consolidator (Structural Plasticity)
        self.sleep_manager = SleepConsolidator(
            substrate=self.core
        )
        
        # 互換性エイリアス
        self.cortex = self 
        
    def _init_core_brain(self):
        model_conf = self.config.get("model", {})
        arch_type = model_conf.get("architecture_type", "sformer")
        
        logger.info(f"Initializing Artificial Brain with architecture: {arch_type}")
        
        if arch_type == "bio_pc_network":
            layer_sizes = model_conf.get("network", {}).get("layer_sizes", [784, 512, 256, 10])
            self.core = BioPCNetwork(layer_sizes=layer_sizes, config=self.config)
        elif arch_type == "sformer":
            self.core = ScaleAndFireTransformer(
                vocab_size=self.config.get("data", {}).get("vocab_size", 50257),
                d_model=model_conf.get("d_model", 256),
                num_layers=model_conf.get("num_layers", 6)
            )
        else:
            logger.warning(f"Unknown architecture '{arch_type}', defaulting to BioPCNetwork")
            self.core = BioPCNetwork(layer_sizes=[64, 128, 10], config=self.config)
            
        self.core.to(self.device)
        self.thinking_engine = self.core

    def forward(self, sensory_input: torch.Tensor) -> torch.Tensor:
        """
        Sensory Processing & Thought Generation
        """
        sensory_input = sensory_input.to(self.device)
        
        # エネルギーチェック
        if self.astrocyte.current_energy <= 5.0:
            logger.warning("Brain Energy Critically Low! Forcing low-power mode (No-Op).")
            # 出力をゼロまたはノイズにして返す
            return torch.zeros_like(sensory_input) # 次元が合うかは簡易処理

        output = self.core(sensory_input)
        
        # 神経活動のモニタリングとエネルギー消費
        firing_rate = 0.0
        if hasattr(self.core, "get_mean_firing_rate"):
            firing_rate = self.core.get_mean_firing_rate()
        
        # アストロサイトへ通知 (代謝コスト発生)
        self.astrocyte.monitor_neural_activity(firing_rate)
        
        return output

    def set_plasticity(self, active: bool):
        self.plasticity_enabled = active
        if hasattr(self.core, "set_online_learning"):
            self.core.set_online_learning(active)
        else:
            self.core.train(active)

    def sleep(self):
        """睡眠モード: エネルギー回復と記憶の整理"""
        logger.info(">>> Sleep Cycle Initiated <<<")
        self.is_awake = False
        self.set_plasticity(False)
        self.sleep_cycle_count += 1
        
        # 1. 内部状態のリセット
        if hasattr(self.core, "reset_state"):
            self.core.reset_state()
            
        # 2. エネルギー回復と疲労除去
        self.astrocyte.replenish_energy(500.0) # 十分な量を回復
        self.astrocyte.clear_fatigue(50.0)
        
        # 3. 構造的可塑性 (Synaptic Pruning / Genesis)
        # 睡眠中に「刈り込み」を行うことで、脳回路を効率化する (Phase 2 Objective)
        stats = self.sleep_manager.perform_maintenance(self.sleep_cycle_count)
        if stats["pruned"] > 0 or stats["created"] > 0:
            logger.info(f"Sleep Maintenance: Pruned={stats['pruned']}, Created={stats['created']} synapses")

    def wake_up(self):
        logger.info(">>> Wake Up <<<")
        self.is_awake = True
        self.set_plasticity(True)

    def get_metrics(self) -> Dict[str, float]:
        metrics = {}
        if hasattr(self.core, "get_sparsity_loss"):
            metrics["sparsity_loss"] = float(self.core.get_sparsity_loss().item())
        
        # アストロサイトの状態も含める
        metrics["energy"] = self.astrocyte.current_energy
        metrics["fatigue"] = self.astrocyte.fatigue
        
        return metrics

    # --- Legacy API ---
    def run_cognitive_cycle(self, text_or_input: Union[str, torch.Tensor]) -> Dict[str, Any]:
        if isinstance(text_or_input, torch.Tensor):
            out = self.forward(text_or_input)
            return {"output": out, "executed_modules": ["Core"], "consciousness": "Active"}
        
        logger.info(f"Cognitive Cycle Input: {text_or_input}")
        if isinstance(self.core, ScaleAndFireTransformer):
            dummy_ids = torch.tensor([[ord(c) % 256 for c in str(text_or_input)[:10]]], dtype=torch.long, device=self.device)
            out = self.core(dummy_ids)
        else:
            # BioPCNetwork用ダミー入力生成
            in_size = 0
            if hasattr(self.core, 'layer_sizes'):
                in_size = self.core.layer_sizes[0]
            elif hasattr(self.core, 'input_size'):
                in_size = self.core.input_size
            else:
                in_size = 64
                
            dummy_tensor = torch.randn(1, in_size, device=self.device)
            out = self.core(dummy_tensor)
            
        return {
            "output": out,
            "executed_modules": ["Core"],
            "consciousness": "Active (Simulated)",
            "thought": str(text_or_input)
        }
        
    def get_brain_status(self) -> Dict[str, Any]:
        report = self.astrocyte.get_diagnosis_report()
        return {
            "astrocyte": report,
            "state": "AWAKE" if self.is_awake else "SLEEPING"
        }

    def sleep_cycle(self):
        self.sleep()
        self.wake_up()