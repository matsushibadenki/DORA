# ファイルパス: snn_research/cognitive_architecture/neuromorphic_scheduler.py
# 日本語タイトル: Neuromorphic OS Kernel Scheduler v2.1 (Type Safe)
# 目的・内容:
#   Type hintsの修正と、get_current_phaseメソッドの追加。

import logging
import heapq
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Union
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace

logger = logging.getLogger(__name__)

class ProcessPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    BACKGROUND = 3
    IDLE = 4

class ResourceLock(Enum):
    WEIGHT_UPDATE = auto()
    SENSORY_INPUT = auto()
    MOTOR_OUTPUT = auto()

@dataclass(order=True)
class SNNProcess:
    priority: int
    name: str = field(compare=False)
    required_locks: Set[ResourceLock] = field(default_factory=set, compare=False)
    energy_cost: float = field(default=1.0, compare=False)
    callback: Any = field(default=None, compare=False)
    
    def __repr__(self):
        return f"<Process '{self.name}' Pr:{self.priority}>"

class NeuromorphicScheduler:
    def __init__(self, astrocyte: AstrocyteNetwork, global_workspace: GlobalWorkspace):
        self.astrocyte = astrocyte
        self.global_workspace = global_workspace
        self.current_phase = "wake"
        self.phase_duration = 0
        
        self.task_queue: List[SNNProcess] = []
        self.active_locks: Set[ResourceLock] = set()
        self.running_processes: List[str] = []
        self.total_tasks_executed = 0
        self.dropped_tasks = 0

    def register_process(self, 
                         name: str, 
                         priority: ProcessPriority, 
                         callback: Any, 
                         required_locks: Optional[List[ResourceLock]] = None,
                         energy_cost: float = 5.0):
        if required_locks is None:
            required_locks = []
            
        process = SNNProcess(
            priority=priority.value,
            name=name,
            required_locks=set(required_locks),
            energy_cost=energy_cost,
            callback=callback
        )
        heapq.heappush(self.task_queue, process)
        logger.debug(f"📋 Task Registered: {name} (Priority: {priority.name})")

    def step(self) -> List[Dict[str, Any]]:
        self.phase_duration += 1
        logs: List[Dict[str, Any]] = []
        
        metrics = self.astrocyte.get_diagnosis_report()["metrics"]
        self._update_system_phase(metrics, logs)
        
        current_energy = metrics.get("current_energy", metrics.get("energy", 0.0))
        max_energy = metrics.get("max_energy", 100.0)
        
        energy_ratio = current_energy / (max_energy + 1e-6)
        
        if energy_ratio > 0.8:
            budget_ratio = 0.5
            mode = "BURST"
        elif energy_ratio > 0.5:
            budget_ratio = 0.3
            mode = "NORMAL"
        else:
            budget_ratio = 0.1
            mode = "SAFETY"

        available_energy_budget = current_energy * budget_ratio
        
        executed_this_step: List[Dict[str, str]] = []
        self.active_locks.clear() 
        
        if self.current_phase == "sleep":
            self.active_locks.add(ResourceLock.SENSORY_INPUT)

        temp_queue = []
        
        while self.task_queue and available_energy_budget > 0:
            process = heapq.heappop(self.task_queue)
            
            if not process.required_locks.isdisjoint(self.active_locks):
                temp_queue.append(process)
                continue
                
            if available_energy_budget < process.energy_cost:
                if process.priority <= ProcessPriority.HIGH.value:
                    self.astrocyte.consume_energy(process.energy_cost * 1.5) 
                    logs.append({"event": "forced_execution", "process": process.name, "reason": "high_priority"})
                    available_energy_budget -= process.energy_cost 
                else:
                    self.dropped_tasks += 1
                    logs.append({
                        "event": "task_dropped", 
                        "process": process.name, 
                        "reason": f"insufficient_budget ({mode})",
                        "budget": available_energy_budget,
                        "cost": process.energy_cost
                    })
                    continue
            else:
                available_energy_budget -= process.energy_cost
                self.astrocyte.consume_energy(process.energy_cost)
            
            try:
                self.active_locks.update(process.required_locks)
                result = None
                if process.callback:
                    result = process.callback()
                
                executed_this_step.append({
                    "process": process.name,
                    "status": "success",
                    "result": str(result)[:50] 
                })
                self.total_tasks_executed += 1
                
            except Exception as e:
                logger.error(f"❌ Process Execution Failed: {process.name} - {e}")
                logs.append({"event": "process_error", "name": process.name, "error": str(e)})

        for p in temp_queue:
            heapq.heappush(self.task_queue, p)
            
        self.running_processes = [log["process"] for log in executed_this_step]
        
        if executed_this_step:
            logs.append({
                "event": "scheduler_step", 
                "executed": self.running_processes, 
                "phase": self.current_phase, 
                "mode": mode
            })

        return logs

    def _update_system_phase(self, metrics: Dict[str, Any], logs: List[Dict[str, Any]]):
        current_energy = metrics.get("energy", metrics.get("current_energy", 0.0))
        max_energy = metrics.get("max_energy", 100.0)
        fatigue = metrics.get("fatigue", 0.0)
        fatigue_threshold = metrics.get("fatigue_threshold", 100.0)

        energy_ratio = current_energy / (max_energy + 1e-6)
        fatigue_ratio = fatigue / (fatigue_threshold + 1e-6)

        if self.current_phase == "wake":
            if energy_ratio < 0.15 or fatigue_ratio > 0.9:
                self._transition_to("sleep")
                logs.append({"event": "phase_change", "to": "sleep", "reason": "exhaustion"})
        
        elif self.current_phase == "sleep":
            if energy_ratio > 0.95 and fatigue_ratio < 0.05:
                self._transition_to("wake")
                logs.append({"event": "phase_change", "to": "wake", "reason": "recovered"})

    def _transition_to(self, new_phase: str):
        logger.info(f"🔄 OS Phase Transition: {self.current_phase} -> {new_phase}")
        self.current_phase = new_phase
        self.phase_duration = 0
        if new_phase == "sleep":
            self._prune_low_priority_tasks()

    def _prune_low_priority_tasks(self):
        initial_len = len(self.task_queue)
        self.task_queue = [p for p in self.task_queue if p.priority <= ProcessPriority.NORMAL.value]
        heapq.heapify(self.task_queue)
        pruned = initial_len - len(self.task_queue)
        if pruned > 0:
            logger.info(f"🧹 Pruned {pruned} background tasks due to sleep transition.")

    def get_status(self) -> Dict[str, Any]:
        return {
            "phase": self.current_phase,
            "queue_depth": len(self.task_queue),
            "running_processes": self.running_processes,
            "total_executed": self.total_tasks_executed,
            "dropped": self.dropped_tasks
        }

    def get_current_phase(self) -> str:
        """実験スクリプト互換用"""
        return self.current_phase