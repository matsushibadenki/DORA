# ファイルパス: snn_research/cognitive_architecture/sleep_consolidation.py
# 日本語タイトル: Sleep Consolidation & Learning Manager
# 目的・内容:
#   - 睡眠フェーズにおける脳の物理的・論理的メンテナンス。
#   - Synaptic Pruning (物理的な結合の削除)
#   - Memory Consolidation (LoRA等による記憶の定着)

import torch
import torch.nn as nn
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class SleepConsolidator:
    """
    Manages structural plasticity and memory consolidation during sleep cycles.
    """

    def __init__(self, substrate: Optional[nn.Module] = None, **kwargs: Any):
        if substrate is None:
            substrate = kwargs.get("target_brain_model")

        if substrate is None:
            logger.warning("SleepConsolidator initialized without substrate!")
            substrate = nn.Module()

        self.substrate = substrate

    def perform_sleep_cycle(self, cycle_count: int = 1, duration_cycles: Optional[int] = None) -> Dict[str, int]:
        """Legacy alias for perform_maintenance."""
        cycles = duration_cycles if duration_cycles is not None else cycle_count
        return self.perform_maintenance(cycles)

    def perform_maintenance(self, cycle_count: int) -> Dict[str, int]:
        """
        睡眠中のメンテナンスを実行する。
        """
        stats = {"pruned": 0, "created": 0, "learned_samples": 0}

        # 1. 物理的メンテナンス (10サイクルに1回実行)
        if cycle_count % 10 == 0:
            stats["pruned"] = self._synaptic_pruning()
            stats["created"] = self._synaptogenesis()

        # 2. 論理的メンテナンス (記憶の学習)
        # 毎回実行、もしくはエネルギーに余裕がある時に実行
        stats["learned_samples"] = self._run_lora_training()

        return stats

    def _synaptic_pruning(self, threshold: float = 0.05) -> int:
        """弱いシナプス結合を物理的に切断（ゼロ化）する"""
        pruned_count = 0
        for name, param in self.substrate.named_parameters():
            if "weight" in name and param.dim() > 1:
                # 重みの絶対値が閾値以下のものをマスク
                mask = torch.abs(param.data) > threshold

                # 統計
                total_synapses = param.numel()
                current_active = int(
                    (torch.abs(param.data) > 1e-6).sum().item())
                new_active = int(mask.sum().item())
                pruned_count += (current_active - new_active)

                # 適用
                param.data *= mask.float()
        return pruned_count

    def _synaptogenesis(self, birth_rate: float = 0.01) -> int:
        """接続されていない箇所に新しいシナプスをランダムに生成する"""
        created_count = 0
        for name, param in self.substrate.named_parameters():
            if "weight" in name and param.dim() > 1:
                # 現在接続がない箇所 (Zero weights)
                zero_mask = (torch.abs(param.data) < 1e-6)

                # 生成確率に基づくマスク
                birth_mask = (torch.rand_like(param.data)
                              < birth_rate) & zero_mask

                # 新しい重みの初期化（小さなランダム値）
                new_connections = torch.randn_like(param.data) * 0.1

                # 適用
                param.data += new_connections * birth_mask.float()

                created_count += int(birth_mask.sum().item())
        return created_count

    def _run_lora_training(self) -> int:
        """
        [NEW] 睡眠学習プロセス (Dreaming)
        短期記憶バッファから重要なエピソードを取り出し、LoRAアダプタ等に追加学習を行う。
        現在は動作検証用のモック実装。
        """
        logger.info("💤 Dreaming... (Running background learning task)")
        
        # 本来はここで:
        # 1. Hippocampus.get_replay_buffer()
        # 2. Loss計算とBackward()
        # 3. Optimizer.step()
        
        # 処理時間をシミュレート（深い眠り）
        time.sleep(0.1) 
        
        # 学習したサンプル数を返す
        return 16 # Batch size dummy