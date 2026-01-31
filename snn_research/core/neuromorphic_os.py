# ファイルパス: snn_research/core/neuromorphic_os.py
# 日本語タイトル: Neuromorphic OS Kernel v1.1
# 目的・内容:
#   - Brainへの委譲プロパティ(device, cycle_count)を追加。
#   - Legacy API (run_cycle, sys_sleep) のサポート。

import logging
import time
import psutil
import asyncio
from typing import Dict, Any, Optional, Union
import torch
from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain

logger = logging.getLogger(__name__)

class NeuromorphicOS:
    """
    Neuromorphic Operating System Kernel.
    Manages the lifecycle, resources, and scheduling of the Artificial Brain.
    """

    def __init__(self, brain: ArtificialBrain, tick_rate: float = 10.0):
        self.brain = brain
        self.tick_rate = tick_rate  # Hz
        self.is_running = False
        self.system_stats: Dict[str, Any] = {}
        
        logger.info(f"🖥️ Neuromorphic OS Kernel initialized. Tick Rate: {tick_rate}Hz")

    # --- Properties delegating to brain (for backward compatibility) ---
    @property
    def device(self) -> torch.device:
        return self.brain.device

    @property
    def cycle_count(self) -> int:
        return self.brain.sleep_cycle_count

    # --- Kernel Methods ---

    def boot(self):
        """システム起動"""
        logger.info(">>> Booting Neuromorphic OS... <<<")
        self.is_running = True
        self.brain.wake_up()
        self._monitor_resources()

    def shutdown(self):
        """システム停止"""
        logger.info(">>> Shutting down Neuromorphic OS... <<<")
        self.is_running = False
        if self.brain.is_awake:
            self.brain.sleep()

    def run_loop(self, duration_sec: Optional[float] = None):
        """メインループを実行"""
        if not self.is_running:
            self.boot()

        start_time = time.time()
        
        try:
            while self.is_running:
                loop_start = time.time()
                self._monitor_resources()
                
                # 自動睡眠制御
                status = self.brain.get_brain_status()
                if status["state"] == "AWAKE" and status["energy"] < 20.0:
                    logger.warning("📉 Low Battery! Initiating emergency sleep cycle.")
                    self.brain.sleep()
                
                if duration_sec and (time.time() - start_time > duration_sec):
                    break
                
                elapsed = time.time() - loop_start
                sleep_time = max(0.0, (1.0 / self.tick_rate) - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Keyboard Interrupt detected.")
        finally:
            self.shutdown()

    def submit_task(self, task_input: Any) -> Dict[str, Any]:
        """外部タスクの受付"""
        if not self.is_running:
            self.boot()
        if not self.brain.is_awake:
            self.brain.wake_up()
        return self.brain.process_step(task_input)

    # --- Legacy API Support ---
    
    def run_cycle(self, sensory_input: Any, phase: str = "wake") -> Dict[str, Any]:
        """
        Legacy: run_cycle wrapper.
        """
        # phase引数はprocess_stepでは現在無視されるが、IF維持のため受け取る
        return self.brain.process_step(sensory_input)

    async def sys_sleep(self, duration: float = 1.0) -> None:
        """
        Legacy: Async sleep wrapper for OmegaPoint.
        """
        logger.info(f"💤 SYS_SLEEP triggered via OS Kernel ({duration}s)")
        self.brain.sleep()
        await asyncio.sleep(duration)
        self.brain.wake_up()

    def _monitor_resources(self):
        process = psutil.Process()
        mem_info = process.memory_info()
        
        self.system_stats = {
            "phys_memory_mb": mem_info.rss / 1024 / 1024,
            "cpu_percent": psutil.cpu_percent(interval=None),
            "brain_energy": self.brain.astrocyte.current_energy,
            "brain_fatigue": self.brain.astrocyte.fatigue
        }

    def get_status_report(self) -> Dict[str, Any]:
        return {
            "os_status": "RUNNING" if self.is_running else "STOPPED",
            "system_resources": self.system_stats,
            "brain_status": self.brain.get_brain_status()
        }