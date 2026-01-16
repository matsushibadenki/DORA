# ファイルパス: snn_research/cognitive_architecture/neuromorphic_scheduler.py
# 日本語タイトル: Neuromorphic Scheduler
# 目的・内容:
#   Neuromorphic OSのリソース管理ユニット。
#   各プロセスの「エネルギー入札（Bid）」に基づき、実行優先順位を決定する。
#   生物学的なアストロサイト（Astrocyte）による代謝制御を模倣。

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
    priority: float  # 0.0 - 1.0
    bid_amount: float # Energy cost
    intent: str

@dataclass(order=True)
class BrainProcess:
    """脳内で実行されるタスク（プロセス）の定義"""
    priority: float # 優先度 (heapqは最小値を取り出すため符号反転して管理)
    name: str = field(compare=False)
    bid_amount: float = field(compare=False) # エネルギー入札額
    callback: Callable = field(compare=False) # 実行する関数
    args: tuple = field(default=(), compare=False)
    is_interrupt: bool = field(default=False, compare=False) # 割り込みフラグ

class NeuromorphicScheduler:
    """
    脳型OSのカーネルスケジューラ。
    """
    def __init__(self, astrocyte_ref: Any, workspace_ref: Optional[Any] = None):
        self.astrocyte = astrocyte_ref
        self.workspace = workspace_ref
        
        # 実行待ちキュー (Priority Queue)
        self.process_queue: List[BrainProcess] = []
        
        # Simulation用: 登録されたプロセスリスト
        self.registered_processes: List[Any] = []
        
        # 実行履歴
        self.execution_log: List[str] = []
        
        logger.info("⚖️ Neuromorphic Scheduler initialized.")

    def register_process(self, process: Any):
        """Simulation用: プロセス定義を登録する"""
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
        # 最終的な優先度スコアの計算
        final_score = (base_priority * energy_bid) if not is_interrupt else 9999.0
        
        # heapqは最小値を取り出すため、スコアをマイナスにして格納
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
        """
        # 1. Simulation Mode: Bid Collection
        if self.registered_processes and input_data is not None:
            context = {"energy": getattr(self.astrocyte, "current_energy", 100.0), "consciousness": None}
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
                            is_interrupt=(bid.priority >= 1.0)
                        )

        # 2. Execution Loop
        results = []
        executed_cost = 0.0
        cycle_budget = 50.0 
        
        # 抑制状態の確認 (Astrocyteからのフィードバック)
        if hasattr(self.astrocyte, "get_diagnosis_report"):
            diagnosis = self.astrocyte.get_diagnosis_report()
            inhibition = diagnosis["metrics"].get("inhibition_level", 0.0)
        else:
            inhibition = 0.0
        
        while self.process_queue:
            process = self.process_queue[0]
            
            # 過剰な活動に対する抑制チェック
            if inhibition > 0.8 and not process.is_interrupt:
                heapq.heappop(self.process_queue)
                logger.debug(f"🚫 Task {process.name} suppressed by Global Inhibition.")
                continue

            # リソース承認
            if self.astrocyte.request_resource(process.name, process.bid_amount):
                heapq.heappop(self.process_queue)
                try:
                    logger.debug(f"▶️ Executing: {process.name}")
                    result = process.callback(*process.args)
                    results.append({"name": process.name, "result": result, "status": "success"})
                    executed_cost += process.bid_amount
                except Exception as e:
                    logger.error(f"❌ Task Execution Failed ({process.name}): {e}")
                    results.append({"name": process.name, "error": str(e), "status": "failed"})
                
                self.execution_log.append(process.name)
                
                if executed_cost >= cycle_budget:
                    break
            else:
                logger.warning(f"⚠️ Resource denied for {process.name}. Scheduler stopping cycle.")
                break
        
        return results

    def clear_queue(self):
        """キューをクリア"""
        self.process_queue = []