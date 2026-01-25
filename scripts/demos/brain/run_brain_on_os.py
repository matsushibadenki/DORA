# ファイルパス: scripts/demos/brain/run_brain_on_os.py
# 日本語タイトル: Neuromorphic OS Integration with Brain v16 (Demo Fixed V2)
# 目的・内容:
#   OSカーネル（Scheduler）上でBrain v16コンポーネントを動作させる統合デモ。
#   - エネルギーレベル20%設定による「覚醒時の省エネモード（反射優先）」を実証する。
#   - Perception呼び出しエラーを修正。

import sys
import os
import torch
import logging
import time
import numpy as np

# パス設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# --- Imports ---
from snn_research.cognitive_architecture.neuromorphic_scheduler import (
    NeuromorphicScheduler, ProcessPriority, ResourceLock
)
from snn_research.utils.observer import NeuromorphicObserver
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
from snn_research.cognitive_architecture.hybrid_perception_cortex import HybridPerceptionCortex
from snn_research.cognitive_architecture.basal_ganglia import BasalGanglia
from snn_research.modules.reflex_module import ReflexModule

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', force=True)
logger = logging.getLogger("BrainOS")

# --- Mock Components for Demo ---

class DemoAstrocyte:
    """デモ用に状態を完全制御できるアストロサイト"""
    def __init__(self):
        self.energy = 100.0
        self.max_energy = 100.0
        self.fatigue = 0.0
        self.fatigue_threshold = 100.0
    
    def get_diagnosis_report(self):
        return {
            "metrics": {
                "energy": self.energy,
                "current_energy": self.energy,
                "max_energy": self.max_energy,
                "fatigue": self.fatigue,
                "fatigue_threshold": self.fatigue_threshold
            }
        }
    
    def consume_energy(self, amount):
        self.energy = max(0.0, self.energy - amount)
        self.fatigue = min(self.fatigue_threshold, self.fatigue + amount * 0.1)

class MockReasoningEngine:
    """高コストな推論エンジンのモック"""
    def __init__(self, device):
        self.device = device
    
    def forward(self, workspace_content):
        time.sleep(0.05) 
        if isinstance(workspace_content, torch.Tensor):
            content_str = f"Tensor shape {workspace_content.shape}"
        else:
            content_str = str(workspace_content)
        thought = f"Analyzed: {content_str}..."
        return {"thought": thought, "confidence": 0.95}

class MockWorldModel:
    def predict(self, action):
        return {"predicted_outcome": "safe"}

# --- Brain Integration Class ---

class NeuromorphicBrainOS:
    """
    OSスケジューラを用いて脳コンポーネントを駆動するラッパー
    """
    def __init__(self, device="cpu"):
        self.device = device
        self.observer = NeuromorphicObserver(experiment_name="brain_on_os_v16")
        
        # 1. 共有リソース
        self.workspace = GlobalWorkspace(dim=256)
        
        # 2. エネルギー管理 (Demo用を使用)
        self.astrocyte = DemoAstrocyte() 
        logger.info("⚡ DemoAstrocyte initialized (Energy: 100%)")
        
        # 3. カーネル (Layer 0)
        self.scheduler = NeuromorphicScheduler(self.astrocyte, self.workspace)
        
        # 4. 脳機能モジュール (Layer 3)
        logger.info("🧠 Building Brain Components...")
        
        self.perception = HybridPerceptionCortex(self.workspace, num_neurons=784, feature_dim=256)
        self.basal_ganglia = BasalGanglia(self.workspace)
        self.reflex = ReflexModule(input_dim=784, action_dim=10).to(device)
        self.reasoning = MockReasoningEngine(device)
        self.world_model = MockWorldModel()
        
        self.step_count = 0

    def receive_input(self, input_data: torch.Tensor, intent: str = "general"):
        self.current_input = input_data
        self.step_count += 1
        logger.info(f"\n📥 Input Received (Step {self.step_count}): Intent='{intent}'")
        
        # 知覚プロセスを登録
        self.scheduler.register_process(
            name="Perception_Process",
            priority=ProcessPriority.HIGH,
            callback=self._process_perception,
            required_locks=[ResourceLock.SENSORY_INPUT],
            energy_cost=1.0 
        )

    # --- Process Callbacks ---

    def _process_perception(self):
        # 知覚処理 (Forward呼び出し)
        features = self.perception(self.current_input)
        self.workspace.write("sensory_buffer", features)
        
        # 可視化ログ
        heatmap_data = features
        if isinstance(features, dict) and "features" in features:
            heatmap_data = features["features"]
        if hasattr(heatmap_data, 'detach'):
             self.observer.log_heatmap(heatmap_data, "perception_features", self.step_count)
        
        # 次のタスク登録: 意思決定
        self.scheduler.register_process(
            name="Decision_Gating",
            priority=ProcessPriority.HIGH,
            callback=self._process_decision_gating,
            energy_cost=0.5
        )
        return "Perception Complete"

    def _process_decision_gating(self):
        # 意思決定 (System 1 vs System 2)
        energy_status = self.astrocyte.get_diagnosis_report()["metrics"]
        energy_ratio = energy_status["current_energy"] / energy_status["max_energy"]
        
        # エネルギー > 30% なら System 2 を検討
        use_system_2 = energy_ratio > 0.3
        
        if use_system_2:
            logger.info("   ⚖️ Decision: Requesting System 2 (Reasoning)")
            self.scheduler.register_process(
                name="System2_Reasoning",
                priority=ProcessPriority.NORMAL,
                callback=self._process_system2_reasoning,
                required_locks=[ResourceLock.WEIGHT_UPDATE], 
                energy_cost=15.0 # 高コスト
            )
        else:
            logger.info("   ⚡ Decision: System 1 Only (Energy Conserving Mode)")
            self.scheduler.register_process(
                name="System1_Reflex",
                priority=ProcessPriority.CRITICAL,
                callback=self._process_reflex,
                energy_cost=2.0 
            )
        return "Gating Complete"

    def _process_system2_reasoning(self):
        content = self.workspace.read("sensory_buffer")
        result = self.reasoning.forward(content)
        self.workspace.write("reasoning_result", result)
        logger.info(f"   🧠 Reasoning Result: {result.get('thought', 'Done')}")
        return "Reasoning Complete"

    def _process_reflex(self):
        try:
            action = self.reflex.forward(self.current_input)
        except:
            action = "Reflex Action"
        return "Reflex Action Triggered"

    # --- Simulation Control ---

    def run_cycle(self):
        logs = self.scheduler.step()
        
        executed = []
        dropped = []
        for l in logs:
            if l.get("event") == "scheduler_step":
                executed.extend(l.get("executed", []))
            elif l.get("event") == "task_dropped":
                dropped.append(l.get("process"))
        
        if executed:
            logger.info(f"   ✅ Executed: {executed}")
        if dropped:
            logger.warning(f"   🚫 Dropped: {dropped}")
            
        self.observer.snapshot_system_state(self.scheduler.get_status(), {}, self.step_count)

    def finalize(self):
        self.observer.save_results()


# --- Main Execution ---

def main():
    logger.info("============================================================")
    logger.info("🚀 Starting Brain v16 on Neuromorphic OS (Stable)")
    logger.info("============================================================")
    
    try:
        brain_os = NeuromorphicBrainOS(device="cpu")
    except Exception as e:
        logger.error(f"❌ Initialization Failed: {e}", exc_info=True)
        return

    input_tensor = torch.randn(1, 784)

    # --- Scenario 1: High Energy ---
    logger.info("\n🧪 [Scenario 1] High Energy State (100%)")
    brain_os.astrocyte.energy = 100.0 # 確実にセット
    brain_os.receive_input(input_tensor, intent="Deep Thought")
    
    for _ in range(3):
        brain_os.run_cycle()
        time.sleep(0.1)

    # --- Scenario 2: Low Energy (Reflex Mode) ---
    # Note: 15%以下は強制スリープのため、20%に設定して
    # "起きているが System 2 は使わない" 状態を作る
    logger.info("\n🧪 [Scenario 2] Low Energy State (20%) -> Force System 1")
    brain_os.astrocyte.energy = 20.0 
    logger.info(f"   ⚠️ Energy Level set to: {brain_os.astrocyte.energy} (Reflex Mode)")
    
    brain_os.receive_input(input_tensor, intent="Quick Reaction")
    
    for _ in range(3):
        brain_os.run_cycle()
        time.sleep(0.1)

    brain_os.finalize()
    logger.info("\n✅ Demo Completed.")

if __name__ == "__main__":
    main()