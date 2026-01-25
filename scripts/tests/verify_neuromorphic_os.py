# ファイルパス: scripts/tests/verify_neuromorphic_os.py
# 日本語タイトル: Neuromorphic OS Integration Test Suite
# 目的・内容:
#   強化されたSchedulerとObserverの連携動作を確認する。
#   1. リソース競合時のロック機能
#   2. エネルギー枯渇時の優先度制御（アドミッションコントロール）
#   3. 観測データの出力（JSON, Heatmap）
#   をシミュレーションし、OSとしての安定性を検証する。

import sys
import os
import time
import logging
import random
import numpy as np

# プロジェクトルートをパスに追加（実行環境に合わせて調整してください）
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

try:
    from snn_research.cognitive_architecture.neuromorphic_scheduler import (
        NeuromorphicScheduler, ProcessPriority, ResourceLock
    )
    from snn_research.utils.observer import NeuromorphicObserver
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("前回の回答コードが snn_research/ 以下に正しく保存されているか確認してください。")
    sys.exit(1)

# --- Mock Classes for Testing ---

class MockAstrocyte:
    """テスト用の疑似アストロサイト（エネルギー管理）"""
    def __init__(self):
        self.energy = 100.0
        self.max_energy = 100.0
        self.fatigue = 0.0
        self.fatigue_threshold = 100.0

    def get_diagnosis_report(self):
        return {
            "metrics": {
                "energy": self.energy,
                "current_energy": self.energy, # Schedulerの実装に合わせてキーを追加
                "max_energy": self.max_energy,
                "fatigue": self.fatigue,
                "fatigue_threshold": self.fatigue_threshold
            }
        }

    def consume_energy(self, amount):
        self.energy = max(0, self.energy - amount)
        self.fatigue += amount * 0.1

class MockGlobalWorkspace:
    """テスト用の疑似グローバルワークスペース"""
    pass

# --- Test Functions ---

def run_os_simulation():
    # 1. Setup
    print("\n🚀 Initializing Neuromorphic OS Test Environment...")
    
    # ディレクトリ作成（エラー回避）
    os.makedirs("benchmarks/results", exist_ok=True)

    astrocyte = MockAstrocyte()
    workspace = MockGlobalWorkspace()
    
    # 強化されたモジュールのインスタンス化
    observer = NeuromorphicObserver(experiment_name="os_stability_test")
    scheduler = NeuromorphicScheduler(astrocyte, workspace)
    
    # ログ設定
    logger = logging.getLogger("OS_Test")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(sh)

    # --- Scenario 1: Resource Locking (Conflict Resolution) ---
    print("\n🧪 [Test 1] Resource Locking & Context Switching")
    print("   -> 同じリソース(WEIGHT_UPDATE)を要求する2つのタスクを投入します。")
    
    def dummy_learning_task_1():
        print("   ✅ Task 1 (Learning) executed.")
        return "Task 1 Done"

    def dummy_learning_task_2():
        print("   ✅ Task 2 (Learning) executed.")
        return "Task 2 Done"

    # タスク登録
    # NOTE: エネルギーコストを4.0に設定（予算オーバーでドロップされないようにする）
    scheduler.register_process(
        name="STDP_Learning",
        priority=ProcessPriority.NORMAL,
        callback=dummy_learning_task_1,
        required_locks=[ResourceLock.WEIGHT_UPDATE],
        energy_cost=4.0 
    )
    
    scheduler.register_process(
        name="FF_Learning",
        priority=ProcessPriority.NORMAL, 
        callback=dummy_learning_task_2,
        required_locks=[ResourceLock.WEIGHT_UPDATE], # 競合するロック
        energy_cost=4.0
    )

    # Step 1: 最初のタスクが実行され、ロック競合で2つ目は待機するはず
    print("\n--- Scheduler Step 1 ---")
    logs = scheduler.step()
    
    executed_processes = []
    dropped_info = []
    for l in logs:
        if l.get("event") == "scheduler_step":
            executed_processes.extend(l.get("executed", []))
        if l.get("event") == "task_dropped":
            dropped_info.append(l)

    print(f"   Executed: {executed_processes}")
    if dropped_info:
        print(f"   ⚠️ Dropped: {dropped_info}")
    
    # 状態スナップショット保存
    observer.snapshot_system_state(scheduler.get_status(), {}, step=1)
    
    # Step 2: 待機していたタスクが実行されるはず
    print("\n--- Scheduler Step 2 ---")
    logs = scheduler.step()
    executed_processes = []
    dropped_info = []
    
    for l in logs:
        if l.get("event") == "scheduler_step":
            executed_processes.extend(l.get("executed", []))
        if l.get("event") == "task_dropped":
            dropped_info.append(l)

    print(f"   Executed: {executed_processes}")
    if dropped_info:
        print(f"   ⚠️ Dropped: {dropped_info}")
    
    # --- Scenario 2: Admission Control (Energy Shortage) ---
    print("\n🧪 [Test 2] Admission Control under Low Energy")
    print("   -> エネルギーを枯渇させ、低優先度タスクが棄却されるか確認します。")
    
    # エネルギーを強制的に下げる
    astrocyte.energy = 5.0 
    print(f"   ⚠️ Current Energy set to: {astrocyte.energy} (CRITICAL)")

    def high_priority_task():
        print("   ✅ High Priority Task executed.")
    
    def low_priority_task():
        print("   ❌ Low Priority Task executed (Unexpected!).")

    scheduler.register_process(
        name="Emergency_Reflex",
        priority=ProcessPriority.CRITICAL, # 緊急
        callback=high_priority_task,
        energy_cost=4.0
    )

    scheduler.register_process(
        name="Background_Dream",
        priority=ProcessPriority.BACKGROUND, # 低優先
        callback=low_priority_task,
        energy_cost=20.0
    )

    print("\n--- Scheduler Step 3 (Low Energy) ---")
    logs = scheduler.step()
    status = scheduler.get_status()
    print(f"   Dropped Tasks Total: {status['dropped']}")
    
    # Observerにイベント記録
    if status['dropped'] > 0:
        observer.log_event("task_dropped", {"count": status['dropped'], "reason": "low_energy"}, step=3)

    # --- Scenario 3: Advanced Observation (Heatmap & Reporting) ---
    print("\n🧪 [Test 3] Visualization & Reporting")
    print("   -> 脳活動のヒートマップと、実験レポートを生成します。")

    # 疑似的な脳活動データ（ランダム行列）
    brain_activity = np.random.rand(20, 20)
    observer.log_heatmap(brain_activity, name="cortex_activity", step=3)
    print("   📸 Heatmap saved.")
    
    # ダッシュボードデータ生成
    observer.generate_dashboard_data()
    observer.save_results()
    print("   💾 Dashboard data & Metrics saved.")

    print(f"\n✅ All Tests Completed. Results are in: {observer.save_dir}")
    print(f"   - Check 'plots/heatmaps/' for visualizations.")
    print(f"   - Check 'system_events.json' for scheduler decisions.")

if __name__ == "__main__":
    run_os_simulation()