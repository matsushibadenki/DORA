# ファイルパス: snn_research/cognitive_architecture/neuromorphic_scheduler.py
# 日本語タイトル: Neuromorphic Scheduler v2.1
# 目的・内容:
#   脳型OSのリソース管理モジュール。
#   Astrocyteからのエネルギー供給状況に基づき、タスク（神経活動）の優先順位制御と実行可否を決定する。

import logging
import heapq
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

@dataclass
class ProcessBid:
    """
    各脳モジュールがスケジューラに対して提出するリソース入札情報。
    """
    module_name: str
    priority: float  # 0.0 - 1.0 (高いほど優先)
    bid_amount: float # 要求エネルギー量
    intent: str

@dataclass(order=True)
class BrainProcess:
    """脳内で実行されるタスク（プロセス）の定義"""
    priority: float # 優先度 (heapqは最小値を取り出すため符号反転して格納)
    name: str = field(compare=False)
    bid_amount: float = field(compare=False) # エネルギー入札額
    callback: Callable = field(compare=False) # 実行する関数
    args: tuple = field(default=(), compare=False)
    is_interrupt: bool = field(default=False, compare=False) # 割り込みフラグ（緊急タスク）

class NeuromorphicScheduler:
    """
    脳型OSのカーネルスケジューラ。
    """
    def __init__(self, astrocyte_ref: Any, workspace_ref: Optional[Any] = None):
        # AstrocyteUnitへの参照を保持
        self.astrocyte = astrocyte_ref
        self.workspace = workspace_ref
        
        # 実行待ちキュー (Priority Queue)
        self.process_queue: List[BrainProcess] = []
        
        # Simulation用: 自動入札を行うプロセス（モジュール）のリスト
        self.registered_processes: List[Any] = []
        
        # 実行履歴（デバッグ・観測用）
        self.execution_log: List[str] = []
        
        logger.info("⚖️ Neuromorphic Scheduler initialized.")

    def register_process(self, process: Any):
        """Simulation用: 定期実行されるプロセス定義を登録する"""
        self.registered_processes.append(process)

    def submit_task(
        self, 
        name: str, 
        callback: Callable, 
        args: tuple = (), 
        base_priority: float = 1.0, 
        energy_bid: float = 10.0,
        is_interrupt: bool = False
    ):
        """
        タスクをスケジューラに登録（入札）する。
        """
        # 最終的な優先度スコアの計算 (優先度 x エネルギー入札額)
        # 緊急タスクは特権的な高スコアを持つ
        final_score = (base_priority * energy_bid) if not is_interrupt else 9999.0
        
        # heapqは最小値popなので、スコアをマイナスにして格納
        process = BrainProcess(
            priority=-final_score,
            name=name,
            bid_amount=energy_bid,
            callback=callback,
            args=args,
            is_interrupt=is_interrupt
        )
        
        heapq.heappush(self.process_queue, process)
        logger.debug(f"📥 Task submitted: {name} (Score: {final_score:.1f}, Bid: {energy_bid})")

    def step(self, input_data: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        1サイクルのスケジューリングと実行を行う。
        Astrocyteの状態に応じて実行可能なタスク数や種類が制限される。
        """
        # 1. Simulation Mode: Registered Processesの自動入札
        if self.registered_processes and input_data is not None:
            # コンテキスト情報の作成
            context = {
                "energy": getattr(self.astrocyte, "current_energy", 100.0),
                "consciousness": None
            }
            if self.workspace:
                context["consciousness"] = self.workspace.get_current_thought()

            for proc in self.registered_processes:
                if hasattr(proc, 'bid_strategy'):
                    bid = proc.bid_strategy(proc.module, input_data, context)
                    if bid.priority > 0:
                        self.submit_task(
                            name=bid.module_name,
                            callback=proc.executor,
                            args=(proc.module, input_data),
                            base_priority=bid.priority,
                            energy_bid=bid.bid_amount,
                            is_interrupt=(bid.priority >= 1.0) # 優先度1.0以上は割り込み扱い
                        )

        # 2. Execution Loop
        results = []
        executed_cost = 0.0
        cycle_budget = 50.0 # 1サイクルあたりの最大処理コスト（仮定値）
        
        # 抑制状態の確認 (Astrocyteからのフィードバック)
        inhibition = 0.0
        if hasattr(self.astrocyte, "get_diagnosis_report"):
            diagnosis = self.astrocyte.get_diagnosis_report()
            inhibition = diagnosis.get("metrics", {}).get("inhibition_level", 0.0)
        
        while self.process_queue:
            # 最も優先度の高いタスクを取り出す
            process = self.process_queue[0]
            
            # 過剰な活動に対する抑制チェック (Global Inhibition)
            # 緊急タスク以外は、抑制レベルが高いと実行キャンセルの可能性がある
            if inhibition > 0.8 and not process.is_interrupt:
                heapq.heappop(self.process_queue)
                logger.debug(f"🚫 Task {process.name} suppressed by Global Inhibition.")
                continue

            # リソース承認 (Astrocyteにエネルギー請求)
            if self.astrocyte.request_resource(process.name, process.bid_amount):
                heapq.heappop(self.process_queue)
                try:
                    # タスク実行
                    logger.debug(f"▶️ Executing: {process.name}")
                    result = process.callback(*process.args)
                    results.append({"name": process.name, "result": result, "status": "success"})
                    executed_cost += process.bid_amount
                except Exception as e:
                    logger.error(f"❌ Task Execution Failed ({process.name}): {e}")
                    results.append({"name": process.name, "error": str(e), "status": "failed"})
                
                self.execution_log.append(process.name)
                
                # サイクルの予算を超えたらループを抜ける（残りのタスクは次サイクルへ持ち越し）
                if executed_cost >= cycle_budget:
                    break
            else:
                # エネルギー不足で拒否された場合、これ以上優先度の低いタスクも実行できない可能性が高いので中断
                logger.warning(f"⚠️ Resource denied for {process.name}. Scheduler stopping cycle.")
                break
        
        return results

    def clear_queue(self):
        """キューをクリア（リセット時など）"""
        self.process_queue = []
        self.execution_log = []