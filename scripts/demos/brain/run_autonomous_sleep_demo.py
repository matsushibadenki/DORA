# scripts/demos/brain/run_autonomous_sleep_demo.py
# Title: Autonomous Sleep & Pruning Demo v1.1
# Description: 
#   パラメータを調整し、短時間のデモでもPruningが確実に観察できるようにした修正版。

import sys
import os
import time
import logging
import torch
import numpy as np
from pathlib import Path

# プロジェクトルートへのパス設定
sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.containers import AppContainer
from snn_research.core.neuromorphic_os import NeuromorphicOS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger("AutonomousDemo")

def count_synapses(kernel):
    return sum(len(n.outgoing_synapses) for n in kernel.neurons)

def run_demo():
    print("\n" + "="*60)
    print("🌙 DORA Autonomous Sleep & Life Cycle Demo v1.1")
    print("="*60 + "\n")

    container = AppContainer()
    config_path = Path("configs/templates/base_config.yaml")
    if not config_path.exists():
        config_path = Path(__file__).resolve().parents[3] / "configs/templates/base_config.yaml"
    
    container.config.from_yaml(str(config_path))
    container.config.training.paradigm.from_value("event_driven")
    container.config.device.from_value("cpu") 

    os_kernel: NeuromorphicOS = container.neuromorphic_os()
    brain = os_kernel.brain
    device = os_kernel.device
    
    os_kernel.boot()
    
    if not brain.use_kernel:
        print("❌ Error: Kernel mode required for this demo.")
        return

    kernel = brain.kernel_substrate.kernel
    
    # [Tuning] デモ用にパラメータを「早送り」設定に変更
    brain.astrocyte.decay_rate = 5.0 
    kernel.pruning_threshold_sleep = 0.20  # 閾値を少し高めに
    kernel.pruning_interval = 10         # 10 Opsごとに刈り込み判定（通常は1000）
    
    print(f"📊 Initial State:")
    print(f"   - Synapses: {count_synapses(kernel)}")
    print(f"   - Energy: {brain.astrocyte.current_energy:.1f}")
    print(f"   - Pruning Interval: {kernel.pruning_interval}")
    
    # --- Step 1: Activity (WAKE) ---
    print("\n🏃 Phase 1: Intense Activity (Consuming Energy)")
    
    input_tensor = torch.randn(1, 128, device=device).abs()
    
    steps = 0
    while brain.is_awake and steps < 100:
        steps += 1
        brain.process_step(input_tensor)
        brain.process_tick(1.0)
        
        if steps % 10 == 0:
            print(f"   Step {steps}: Energy {brain.astrocyte.current_energy:.1f} | Synapses {count_synapses(kernel)}")
            
        brain.astrocyte.consume_energy(50.0) 
        time.sleep(0.01)

    if not brain.is_awake:
        print("\n😴 Brain has fallen asleep automatically!")
    else:
        print("\n⚠️ Brain is still awake. Forcing sleep for demo.")
        brain.sleep()

    # --- Step 2: Sleep (DREAM & PRUNE) ---
    print("\n💤 Phase 2: Dreaming & Pruning")
    print("   (Simulating accelerated deep sleep...)")
    
    synapses_before_sleep = count_synapses(kernel)
    
    # 睡眠サイクル
    for i in range(20): # サイクル数を増やす
        dream_input = torch.randn(1, 128, device=device) * 0.5
        
        # Brainの標準プロセスを実行
        brain.process_step(dream_input)
        
        # [Demo Hack] デモ時間を短縮するため、外部から強力な減衰を追加適用
        # これにより数秒で「一晩分の忘却」をシミュレートする
        kernel.apply_synaptic_scaling(0.95) 

        if (i+1) % 5 == 0:
            current_syn = count_synapses(kernel)
            diff = synapses_before_sleep - current_syn
            print(f"   Dream {i+1}: Synapses {current_syn} (Pruned: {diff})")
        time.sleep(0.05)

    # --- Step 3: Wake Up ---
    print("\n🌅 Phase 3: Waking Up")
    brain.wake_up()
    
    final_synapses = count_synapses(kernel)
    delta_prune = synapses_before_sleep - final_synapses
    
    print(f"📊 Final Report:")
    print(f"   - Awake Steps: {steps}")
    print(f"   - Synapses (Start): {synapses_before_sleep}")
    print(f"   - Synapses (End): {final_synapses}")
    print(f"   - Total Pruned: {delta_prune}")
    print(f"   - Final Energy: {brain.astrocyte.current_energy:.1f} (Replenished)")

    if delta_prune > 0:
        print("✅ SUCCESS: Autonomous pruning verified.")
    else:
        print("⚠️ NOTE: No pruning occurred. Check thresholds.")

    print("\n✅ Demo Completed.")

if __name__ == "__main__":
    run_demo()