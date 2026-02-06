# scripts/experiments/brain/run_lifelong_learning_test.py
# Title: Lifelong Learning Verification (Sleep Effect)
# Description: 
#   睡眠（シナプス刈り込み）が「壊滅的忘却」の防止にどう寄与するかを検証する実験。
#   Task A学習 -> 睡眠 -> Task B学習 -> Task Aの記憶テスト という順序で実行。

import sys
import os
import time
import logging
import torch
import numpy as np
from pathlib import Path

# プロジェクトルート設定
sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.containers import AppContainer
from snn_research.core.neuromorphic_os import NeuromorphicOS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger("LifelongExp")

def generate_task_data(device, pattern_id=0, size=10):
    """特定のパターン（タスク）の入出力を生成"""
    torch.manual_seed(pattern_id) # タスクごとに固定シード
    inputs = torch.randn(size, 128, device=device).abs()
    # シンプルなマッピング: 入力の一部に強く反応するターゲットを作る
    targets = (inputs[:, :10].sum(dim=1, keepdim=True) > 5.0).float()
    return inputs, targets

def train_brain(brain, inputs, label, steps=20):
    """簡易的な学習ループ（Hebbian/STDP的な活性化）"""
    logger.info(f"📚 Training Task {label}...")
    start_energy = brain.astrocyte.current_energy
    
    for i in range(steps):
        # バッチ処理的に入力を流す
        for x in inputs:
            brain.process_step(x.unsqueeze(0))
    
    logger.info(f"   -> Finished Task {label}. Energy consumed: {start_energy - brain.astrocyte.current_energy:.1f}")

def test_brain(brain, inputs, label):
    """記憶の強度をテスト（出力の安定性や発火強度で測定）"""
    total_response = 0.0
    with torch.no_grad():
        for x in inputs:
            res = brain.process_step(x.unsqueeze(0))
            out = res.get("output")
            if out is not None:
                total_response += out.mean().item()
    
    avg_resp = total_response / len(inputs)
    logger.info(f"📝 Test Task {label}: Mean Response = {avg_resp:.4f}")
    return avg_resp

def run_experiment():
    print("\n" + "="*60)
    print("🧠 DORA Lifelong Learning Experiment: The Sleep Benefit")
    print("="*60 + "\n")

    container = AppContainer()
    config_path = Path("configs/templates/base_config.yaml")
    if not config_path.exists():
        config_path = Path(__file__).resolve().parents[3] / "configs/templates/base_config.yaml"
    
    container.config.from_yaml(str(config_path))
    container.config.training.paradigm.from_value("event_driven")
    container.config.device.from_value("cpu")

    os_kernel = container.neuromorphic_os()
    brain = os_kernel.brain
    device = os_kernel.device
    os_kernel.boot()

    # パラメータ調整
    brain.astrocyte.decay_rate = 1.0
    if brain.use_kernel:
        brain.kernel_substrate.kernel.pruning_threshold_sleep = 0.15
        brain.kernel_substrate.kernel.pruning_interval = 50

    # データ生成
    task_A_in, _ = generate_task_data(device, pattern_id=100) # Task A
    task_B_in, _ = generate_task_data(device, pattern_id=200) # Task B (Aとは異なるパターン)

    # --- Phase 1: Learn Task A ---
    train_brain(brain, task_A_in, "A", steps=5)
    score_A_initial = test_brain(brain, task_A_in, "A (Initial)")
    
    # --- Phase 2: Sleep & Consolidate ---
    print("\n💤 Sleeping to consolidate Task A...")
    brain.sleep()
    
    # 睡眠中の処理（Scaling & Pruning）
    for _ in range(10):
        # 夢（Task Aの再活性化を模倣したノイズ入力）
        dream_input = task_A_in[0].unsqueeze(0) + torch.randn_like(task_A_in[0].unsqueeze(0)) * 0.1
        brain.process_step(dream_input)
        if hasattr(brain.kernel_substrate.kernel, "apply_synaptic_scaling"):
             brain.kernel_substrate.kernel.apply_synaptic_scaling(0.98) # 睡眠中の減衰
        time.sleep(0.05)
        
    brain.wake_up()
    print("🌅 Woke up. Brain is refreshed.")

    # --- Phase 3: Learn Task B ---
    # ここでTask Aを忘れてしまうか（干渉するか）？
    print("\n📚 Learning new Task B (Interference check)...")
    train_brain(brain, task_B_in, "B", steps=5)
    
    # --- Phase 4: Final Test ---
    print("\n📊 Final Evaluation:")
    score_A_final = test_brain(brain, task_A_in, "A (After Task B)")
    score_B_final = test_brain(brain, task_B_in, "B (New Memory)")
    
    retention_rate = (score_A_final / score_A_initial) * 100 if score_A_initial > 0 else 0
    
    print("-" * 30)
    print(f"   Retention of Task A: {retention_rate:.1f}%")
    print(f"   Acquisition of Task B: {score_B_final:.4f}")
    
    if retention_rate > 80.0:
        print("✅ SUCCESS: Catastrophic Forgetting Mitigated!")
        print("   The brain retained old memories while learning new ones.")
    else:
        print("⚠️ WARNING: Some forgetting occurred.")

if __name__ == "__main__":
    run_experiment()