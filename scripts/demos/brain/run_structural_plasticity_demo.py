# scripts/demos/brain/run_structural_plasticity_demo.py
# Title: Structural Plasticity Demo v2.2 (SHY Enhanced)
# Description: 
#   睡眠時のシナプス刈り込みをより確実に発生させるための調整版。
#   閾値を動的に操作し、Synaptic Scalingによる能動的な忘却（Forget）をシミュレートする。

import sys
import os
import time
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.containers import AppContainer
from snn_research.core.neuromorphic_os import NeuromorphicOS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger("PlasticityDemo")

def count_total_synapses(kernel):
    return sum(len(n.outgoing_synapses) for n in kernel.neurons)

def run_demo():
    print("\n" + "="*60)
    print("🌱 DORA Structural Plasticity Demo v2.2: The Cycle of Life")
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
    
    if not brain.use_kernel or not brain.kernel_substrate:
        print("❌ Error: Kernel mode not active.")
        return
    
    kernel = brain.kernel_substrate.kernel
    
    # [Tuning] デモ用にPruning閾値を調整
    kernel.pruning_threshold_sleep = 0.2  # 通常(0.05)より高くして刈り込みやすくする
    
    initial_synapses = count_total_synapses(kernel)
    print(f"📊 Initial Network State:")
    print(f"   - Neurons: {len(kernel.neurons)}")
    print(f"   - Synapses: {initial_synapses}")
    print(f"   - Sleep Pruning Threshold: {kernel.pruning_threshold_sleep}")
    
    # --- Phase 1: WAKE ---
    print("\n🌪️  Phase 1: [WAKE] Intense Learning (Growth)")
    stimulus_A = torch.zeros(1, 128, device=device)
    stimulus_A[0, 20:40] = 5.0 

    for i in range(5):
        os_kernel.submit_task(stimulus_A, synchronous=True)
        created = kernel.stats['synapses_created']
        current = count_total_synapses(kernel)
        print(f"   Ep {i+1}: Synapses {current} (🌱+{created})")
        time.sleep(0.1)

    after_wake_synapses = count_total_synapses(kernel)
    
    # --- Phase 2: SLEEP ---
    print("\n💤 Phase 2: [SLEEP] Consolidation & Pruning")
    print("   Switching to sleep mode. Aggressive pruning activated.")
    print("   Simulating long-term sleep (20 cycles) with Synaptic Scaling...")
    
    os_kernel.shutdown() 
    
    # 睡眠サイクルを長めに回す
    dream_stimulus = torch.rand(1, 128, device=device) * 2.0
    
    for i in range(20):
        # 睡眠中の夢（Replay）処理
        brain.process_step(dream_stimulus)
        
        # [New] 睡眠中のシナプス恒常性維持（SHY）をシミュレート
        # ステップごとに全シナプスをわずかに減衰させ、閾値以下にする
        if hasattr(kernel, "apply_synaptic_scaling"):
            kernel.apply_synaptic_scaling(0.98) # 2% decay per step

        # 5回に1回ログ出力
        if (i+1) % 5 == 0:
            pruned = kernel.stats['synapses_pruned']
            current = count_total_synapses(kernel)
            print(f"   Dream {i+1}: Synapses {current} (✂️-{pruned} cumulative)")
        time.sleep(0.05)
    
    final_synapses = count_total_synapses(kernel)
    delta_growth = after_wake_synapses - initial_synapses
    delta_prune = final_synapses - after_wake_synapses
    
    print("\n📊 Final Report:")
    print(f"   - Initial: {initial_synapses}")
    print(f"   - Peak (Wake): {after_wake_synapses} (+{delta_growth} grown)")
    print(f"   - Final (Sleep): {final_synapses} ({delta_prune} pruned)")
    
    if delta_prune < 0:
        print("   ✅ SUCCESS: Sleep pruning reduced connection count.")
        print("      The brain has forgotten weak memories to save energy.")
    else:
        print("   ⚠️ NOTE: Pruning was minor or did not occur.")
        print("      Try increasing scaling factor or sleep duration.")

    print("\n✅ Demo Completed Successfully.")

if __name__ == "__main__":
    run_demo()