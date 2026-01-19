# ファイルパス: snn_research/cognitive_architecture/sleep_consolidation.py
# 日本語タイトル: Sleep Consolidation & Structural Plasticity Manager
# 目的・内容:
#   睡眠フェーズにおける脳の物理的メンテナンスを行う。
#   - Synaptic Pruning (刈り込み): 弱い結合の削除
#   - Synaptogenesis (生成): 新しい結合のランダム生成
#   - Homeostasis (恒常性): 重みの正規化

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class SleepConsolidator:
    """
    Manages structural plasticity during sleep cycles.
    """

    def __init__(self, substrate: Optional[nn.Module] = None, **kwargs: Any):
        if substrate is None:
            # Fallback for legacy calls passing target_brain_model
            substrate = kwargs.get("target_brain_model")

        if substrate is None:
            # If still None, maybe raise warning or handle gracefully
            # For now, create a dummy module to avoid attribute errors if not critical
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
        stats = {"pruned": 0, "created": 0}

        # 10サイクルに1回実行（頻度調整）
        if cycle_count % 10 != 0:
            return stats

        stats["pruned"] = self._synaptic_pruning()
        stats["created"] = self._synaptogenesis()

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
