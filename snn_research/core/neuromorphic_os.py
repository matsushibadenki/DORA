# ファイルパス: snn_research/core/neuromorphic_os.py
# 日本語タイトル: Neuromorphic OS Kernel v2.0 (Real-Time Life Cycle)
# 目的・内容:
#   - 生物的なリアルタイムループ(Life Cycle)の実装。
#   - 非同期キューを用いた外部入力の受付。
#   - アイドル時の自発的思考(Tick)のトリガー。

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
    """
    Neuromorphic Operating System Kernel.
    Manages the lifecycle, resources, and scheduling of the Artificial Brain.
    Now supports real-time biological constraints.
    """

    def __init__(self, brain: ArtificialBrain, tick_rate: float = 1.0):
        self.brain = brain
        self.tick_rate = tick_rate  # Hz (1秒間に何回思考/代謝チェックを行うか)
        self.is_running = False
        self.system_stats: Dict[str, Any] = {}
        
        # 外部からの刺激（入力）を溜める感覚バッファ
        self.input_queue = queue.Queue()
        
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
        """
        生物的メインループ (Life Cycle Loop)
        ユーザー入力がない時間も、代謝と自発的思考を行い続ける。
        """
        if not self.is_running:
            self.boot()

        start_time = time.time()
        last_tick_time = time.time()
        
        logger.info("❤️ Life Cycle Started. Waiting for inputs or spontaneous thoughts...")

        try:
            while self.is_running:
                loop_start = time.time()
                self._monitor_resources()
                
                # 1. 終了条件の確認
                if duration_sec and (time.time() - start_time > duration_sec):
                    break

                # 2. 自動睡眠制御 (バッテリー切れ)
                status = self.brain.get_brain_status()
                if status["state"] == "AWAKE" and status["energy"] < 10.0:
                    logger.warning("📉 Low Battery! Initiating emergency sleep cycle.")
                    self.brain.sleep_cycle() # 寝て、回復して、起きる

                # 3. 入力処理 vs アイドル処理 (Tick)
                try:
                    # キューから入力を取得 (ブロッキングなし)
                    # 割り込み処理：入力があれば即座に脳に伝える
                    task_input = self.input_queue.get_nowait()
                    logger.info(f"👂 Sensory Input Detected: {str(task_input)[:50]}...")
                    self.brain.process_step(task_input)
                    
                    # 入力があったので、退屈タイマー的なものをリセットする処理をBrain側で行うことを期待
                
                except queue.Empty:
                    # 入力がない場合 -> 時間経過(Tick)を脳に伝える
                    current_time = time.time()
                    delta_time = current_time - last_tick_time
                    
                    # 一定間隔(tick_rate)以上経過していたらTick処理
                    if delta_time >= (1.0 / self.tick_rate):
                        self.brain.process_tick(delta_time)
                        last_tick_time = current_time

                # 4. CPU負荷調整 (Busy Wait防止)
                # ループの回転速度を制御（Tickレートとは別、OSとしての応答性）
                elapsed = time.time() - loop_start
                # 最低でも0.01秒はスリープして他のプロセスにCPUを譲る
                sleep_time = max(0.01, (0.1) - elapsed) 
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Keyboard Interrupt detected.")
        finally:
            self.shutdown()

    def submit_task(self, task_input: Any) -> Dict[str, Any]:
        """
        外部タスクの受付 (非同期)
        ユーザーからの入力を感覚バッファ(Queue)に積む。
        即時実行ではなく、Life Cycleの中で処理される。
        """
        if not self.is_running:
            self.boot()
            
        self.input_queue.put(task_input)
        
        # 呼び出し元には「受け付けました」と返す
        return {
            "status": "queued",
            "message": "Input received by sensory buffer.",
            "queue_size": self.input_queue.qsize()
        }

    # --- Legacy API Support ---
    
    def run_cycle(self, sensory_input: Any, phase: str = "wake") -> Dict[str, Any]:
        """
        Legacy: run_cycle wrapper.
        直接実行したい場合のために残すが、基本はsubmit_task推奨。
        """
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
            "brain_energy": self.brain.astrocyte.current_energy if hasattr(self.brain, 'astrocyte') else 0,
            "brain_fatigue": self.brain.astrocyte.fatigue if hasattr(self.brain, 'astrocyte') else 0
        }

    def get_status_report(self) -> Dict[str, Any]:
        return {
            "os_status": "RUNNING" if self.is_running else "STOPPED",
            "system_resources": self.system_stats,
            "brain_status": self.brain.get_brain_status()
        }