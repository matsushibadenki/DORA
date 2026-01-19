# ファイルパス: snn_research/cognitive_architecture/neuromorphic_scheduler.py
# 日本語タイトル: Homeostatic Scheduler (Meta-Cognition)
# 目的・内容:
#   アストロサイトからの生体シグナルに基づき、システムの「モード（Wake/Sleep）」を切り替える。
#   自律的な行動生成の最上位ループ制御を行う。

import logging
from typing import Dict, Any, List, Optional, Union
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace

logger = logging.getLogger(__name__)


class NeuromorphicScheduler:
    """
    Decides the system's phase (Wake, Sleep, Dream) based on homeostasis.
    """

    def __init__(self, astrocyte: AstrocyteNetwork, global_workspace: GlobalWorkspace):
        self.astrocyte = astrocyte
        self.global_workspace = global_workspace

        self.current_phase = "wake"
        self.phase_duration = 0
        self.task_queue: List[str] = []

    def step(self) -> List[Dict[str, Any]]:
        """
        スケジューリングの実行。
        状態遷移ロジック：
        - Wake -> Sleep: エネルギー枯渇 or 疲労限界
        - Sleep -> Wake: エネルギー満タン and 疲労解消
        """
        self.phase_duration += 1
        logs = []

        report = self.astrocyte.get_diagnosis_report()
        metrics = report["metrics"]

        # Phase Transition Logic
        if self.current_phase == "wake":
            if metrics["energy"] < (metrics["max_energy"] * 0.1) or \
               metrics["fatigue"] > metrics["fatigue_threshold"]:
                self._transition_to("sleep")
                logs.append({"event": "phase_change",
                            "to": "sleep", "reason": "exhaustion"})

        elif self.current_phase == "sleep":
            if metrics["energy"] > (metrics["max_energy"] * 0.9) and \
               metrics["fatigue"] < (metrics["fatigue_threshold"] * 0.1):
                self._transition_to("wake")
                logs.append({"event": "phase_change",
                            "to": "wake", "reason": "recovered"})

        # Update Workspace context based on phase
        if self.current_phase == "sleep":
            # 睡眠中は外部入力を遮断し、内部生成モードへ
            pass

        return logs

    def _transition_to(self, new_phase: str):
        logger.info(f"🔄 Phase Transition: {self.current_phase} -> {new_phase}")
        self.current_phase = new_phase
        self.phase_duration = 0

    def clear_queue(self):
        self.task_queue = []

    def get_current_phase(self) -> str:
        return self.current_phase


class BrainProcess:
    """Mock process for OS simulation compatibility."""

    def __init__(self, name: str, priority: float = 0.5):
        self.name = name
        self.priority = priority


class ProcessBid:
    """Mock bid for OS simulation compatibility."""

    def __init__(self, process: Union[BrainProcess, str], bid_value: float, cost: float = 0.0, intent: str = ""):
        self.process = process
        self.bid_value = bid_value
        self.cost = cost
        self.intent = intent
