# ファイルパス: snn_research/cognitive_architecture/sleep_consolidation.py
# 日本語タイトル: Sleep Consolidation & Learning Manager (SDFT Enhanced)
# 目的・内容:
#   - 睡眠フェーズにおける脳の物理的・論理的メンテナンス。
#   - Synaptic Pruning (物理的な結合の削除)
#   - Memory Consolidation (SDFTによる記憶定着)

import torch
import torch.nn as nn
import logging
import time
import random
from typing import Dict, Any, Optional, List

# [NEW] SDFT用のマネージャーをインポート
# 循環参照を避けるためTYPE_CHECKING等での対応が望ましいが、ここでは簡易的に配置
try:
    from snn_research.distillation.thought_distiller import ThoughtDistillationManager, SymbolicTeacher
except ImportError:
    ThoughtDistillationManager = Any  # type: ignore
    SymbolicTeacher = Any  # type: ignore

logger = logging.getLogger(__name__)


class SleepConsolidator:
    """
    Manages structural plasticity and memory consolidation during sleep cycles.
    Implements SDFT (Self-Distillation Fine-Tuning) for continual learning.
    """

    def __init__(self, substrate: Optional[nn.Module] = None, **kwargs: Any):
        if substrate is None:
            substrate = kwargs.get("target_brain_model")

        if substrate is None:
            logger.warning("SleepConsolidator initialized without substrate!")
            substrate = nn.Module()

        self.substrate = substrate
        
        # SDFT用のコンポーネント初期化
        # 実際にはDIコンテナ(containers.py)から注入されるべき
        self.teacher = SymbolicTeacher()
        self.distiller = ThoughtDistillationManager(self.substrate, self.teacher)
        
        # 短期記憶（デモンストレーション用バッファ）
        self.episodic_buffer: List[Dict[str, Any]] = []

    def add_episode(self, episode: Dict[str, Any]):
        """活動中に得られた良質なエピソード(Input, Chain, Answer)を保存"""
        self.episodic_buffer.append(episode)
        # バッファサイズ制限
        if len(self.episodic_buffer) > 100:
            self.episodic_buffer.pop(0)

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

        # 2. 論理的メンテナンス (SDFT / Memory Consolidation)
        # エピソードが十分に溜まっている場合、夢を見る (Dreaming/SDFT)
        if len(self.episodic_buffer) >= 3:
            stats["learned_samples"] = self._perform_sdft_dreaming()
        else:
            # 従来モード (LoRA Mock)
            stats["learned_samples"] = self._run_lora_training()

        return stats

    def _synaptic_pruning(self, threshold: float = 0.05) -> int:
        """弱いシナプス結合を物理的に切断（ゼロ化）する"""
        pruned_count = 0
        for name, param in self.substrate.named_parameters():
            if "weight" in name and param.dim() > 1:
                mask = torch.abs(param.data) > threshold
                total_synapses = param.numel()
                current_active = int(
                    (torch.abs(param.data) > 1e-6).sum().item())
                new_active = int(mask.sum().item())
                pruned_count += (current_active - new_active)
                param.data *= mask.float()
        return pruned_count

    def _synaptogenesis(self, birth_rate: float = 0.01) -> int:
        """接続されていない箇所に新しいシナプスをランダムに生成する"""
        created_count = 0
        for name, param in self.substrate.named_parameters():
            if "weight" in name and param.dim() > 1:
                zero_mask = (torch.abs(param.data) < 1e-6)
                birth_mask = (torch.rand_like(param.data)
                              < birth_rate) & zero_mask
                new_connections = torch.randn_like(param.data) * 0.1
                param.data += new_connections * birth_mask.float()
                created_count += int(birth_mask.sum().item())
        return created_count

    def _run_lora_training(self) -> int:
        """従来の睡眠学習プロセス (Mock)"""
        # logger.debug("💤 Deep sleep (No dreams)...")
        time.sleep(0.05) 
        return 0

    def _perform_sdft_dreaming(self) -> int:
        """
        [SDFT Implementation] 睡眠中の自己蒸留プロセス (Dreaming)。
        過去のエピソード(Demonstrations)をランダムに選び、それを元に
        新しい問題（仮想的な状況）に対する推論を行い、学習する。
        """
        logger.info("🦄 Dreaming with SDFT (Self-Distillation)...")
        
        # 1. デモンストレーションのサンプリング (過去の記憶)
        demos = random.sample(self.episodic_buffer, min(len(self.episodic_buffer), 3))
        
        # 2. 新しい問題の生成 (ここでは簡易的に、過去の問題の数値をランダムに変形するシミュレーション)
        # 本来はGenerative Modelが新しい問題を生成する
        seed_problem = random.choice(self.episodic_buffer)["input"]
        new_problems = [self._mutate_problem(seed_problem) for _ in range(5)]
        
        # 3. SDFTデータセット生成 (TeacherがICLで解く)
        sdft_dataset = self.distiller.generate_sdft_dataset(new_problems, demos)
        
        # 4. 蒸留 (System 1の更新)
        if sdft_dataset:
            self.distiller.distill(sdft_dataset, epochs=1)
            
        return len(sdft_dataset)

    def _mutate_problem(self, problem: str) -> str:
        """
        デモ用: 算数問題の数値をランダムに変更して新しい問題を生成する。
        Ex: "15 + 27" -> "12 + 30"
        """
        try:
            parts = problem.replace("?", "").split("+")
            a = int(parts[0].strip())
            b = int(parts[1].strip())
            
            new_a = max(1, a + random.randint(-5, 5))
            new_b = max(1, b + random.randint(-5, 5))
            return f"{new_a} + {new_b}"
        except:
            return "10 + 10"