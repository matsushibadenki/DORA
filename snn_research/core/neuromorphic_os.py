# snn_research/core/neuromorphic_os.py
# Title: Neuromorphic OS Kernel v2.3 (System Calls)
# Description: sys_sleep (async) を追加。

import logging
import time
import psutil
import asyncio
import queue
from typing import Dict, Any, Optional, Union
import torch
from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain

logger = logging.getLogger(__name__)

class NeuromorphicOS:
    def __init__(self, brain: ArtificialBrain, tick_rate: float = 1.0):
        self.brain = brain
        self.tick_rate = tick_rate
        self.is_running = False
        self.system_stats: Dict[str, Any] = {}
        self.input_queue: queue.Queue[Any] = queue.Queue()
        
        logger.info(f"🖥️ Neuromorphic OS Kernel initialized. Tick Rate: {tick_rate}Hz")

    @property
    def device(self) -> torch.device:
        return self.brain.device

    @property
    def cycle_count(self) -> int:
        return self.brain.sleep_cycle_count

    def boot(self):
        logger.info(">>> Booting Neuromorphic OS... <<<")
        self.is_running = True
        self.brain.wake_up()
        self._monitor_resources()

    def shutdown(self):
        logger.info(">>> Shutting down Neuromorphic OS... <<<")
        self.is_running = False
        if self.brain.is_awake:
            self.brain.sleep()

    # [Fix] Async sleep system call
    async def sys_sleep(self):
        self.shutdown()
        await asyncio.sleep(0.1)

    def submit_task(self, task_input: Any, synchronous: bool = True) -> Dict[str, Any]:
        if not self.is_running:
            self.boot()
            
        if synchronous:
            start_time = time.time()
            result = self.brain.process_step(task_input)
            result["os_overhead"] = time.time() - start_time
            return result
        else:
            self.input_queue.put(task_input)
            return {
                "status": "queued",
                "message": "Input received by sensory buffer.",
                "queue_size": self.input_queue.qsize()
            }

    def run_cycle(self, task_input: Any, phase: str = "wake") -> Dict[str, Any]:
        return self.submit_task(task_input, synchronous=True)

    def run_loop(self, duration_sec: Optional[float] = None):
        if not self.is_running:
            self.boot()

        start_time = time.time()
        last_tick_time = time.time()
        logger.info("❤️ Life Cycle Started (Async Mode).")

        try:
            while self.is_running:
                loop_start = time.time()
                self._monitor_resources()
                if duration_sec and (time.time() - start_time > duration_sec): break

                try:
                    task_input = self.input_queue.get_nowait()
                    self.brain.process_step(task_input)
                except queue.Empty:
                    current_time = time.time()
                    if current_time - last_tick_time >= (1.0 / self.tick_rate):
                        self.brain.process_tick(current_time - last_tick_time)
                        last_tick_time = current_time

                sleep_time = max(0.01, 0.1 - (time.time() - loop_start))
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def _monitor_resources(self):
        try:
            process = psutil.Process()
            mem_info = process.memory_info()
            self.system_stats = {
                "phys_memory_mb": mem_info.rss / 1024 / 1024,
                "cpu_percent": psutil.cpu_percent(interval=None),
                "brain_energy": self.brain.astrocyte.current_energy if hasattr(self.brain, 'astrocyte') else 0
            }
        except Exception:
            pass

    def get_status_report(self) -> Dict[str, Any]:
        return {
            "os_status": "RUNNING" if self.is_running else "STOPPED",
            "system_resources": self.system_stats,
            "brain_status": self.brain.get_brain_status()
        }