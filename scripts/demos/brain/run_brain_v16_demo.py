# ファイルパス: scripts/demos/brain/run_brain_v16_demo.py
# 日本語タイトル: Integrated Brain v16.10 Integration Demo (Plot Fix)
# 目的・内容:
#   - 睡眠サイクル(shutdown)で履歴が消える前にデータを退避させ、確実にプロットを描画する。

import sys
import os
import time
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.containers import AppContainer
from snn_research.core.neuromorphic_os import NeuromorphicOS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger("BrainDemo")

def generate_visual_stimulus(pattern_type: str = "random", device: torch.device = torch.device("cpu")) -> torch.Tensor:
    if pattern_type == "prey":
        t = torch.zeros(1, 128, device=device)
        t[0, 10:40] = 5.0
        return t
    elif pattern_type == "predator":
        t = torch.zeros(1, 128, device=device)
        t[0, 80:110] = 5.0
        return t
    else:
        return (torch.rand(1, 128, device=device) * 5.0)

def save_raster_plot(spike_history, output_path="runtime_state/brain_raster_plot.png"):
    """スパイク履歴からラスタープロットを作成して保存"""
    if not spike_history:
        logger.warning("⚠️ No spikes to plot.")
        return

    logger.info(f"🎨 Generating Raster Plot ({len(spike_history)} spikes)...")
    
    times = [x[0] for x in spike_history]
    neuron_ids = [x[1] for x in spike_history]
    is_inhibitory = [x[2] for x in spike_history]
    
    colors = ['red' if inh else 'blue' for inh in is_inhibitory]
    
    plt.figure(figsize=(12, 8))
    plt.scatter(times, neuron_ids, s=2, c=colors, alpha=0.6)
    
    plt.title("DORA Brain Activity: Spike Raster Plot")
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron ID")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    logger.info(f"✅ Raster Plot saved to: {output_path}")
    plt.close()

def run_demo():
    print("\n" + "="*60)
    print("🧠 Neuromorphic OS & Artificial Brain v16.10 Integration Demo")
    print("="*60 + "\n")

    container = AppContainer()
    config_path = Path("configs/templates/base_config.yaml")
    if not config_path.exists():
        config_path = Path(__file__).resolve().parents[3] / "configs/templates/base_config.yaml"
    
    container.config.from_yaml(str(config_path))
    
    print("⚙️  Configuring system for Event-Driven SNN mode...")
    container.config.training.paradigm.from_value("event_driven")
    container.config.device.from_value("cpu") 

    os_kernel: NeuromorphicOS = container.neuromorphic_os()
    brain = os_kernel.brain
    device = os_kernel.device

    print(f"✅ System Initialized on {device}")
    status = brain.get_brain_status()
    print(f"   - Execution Mode: {status.get('mode', 'Unknown')}")
    print(f"   - OS Kernel: v2.1 (Tick: {os_kernel.tick_rate}Hz)")
    
    os_kernel.boot()
    
    print("\n🌞 [PHASE 1] WAKE CYCLE - Active Inference & Learning")
    
    episodes = [
        ("predator", "Run away!"),
        ("prey", "Chase it!"),
        ("random", "Ignore"),
        ("predator", "Run away!") 
    ]

    for i, (stimulus_type, expected_intent) in enumerate(episodes):
        print(f"\n⏱️  Episode {i+1}: Encountering '{stimulus_type}'")
        
        visual_input = generate_visual_stimulus(stimulus_type, device)
        start_time = time.time()
        
        result = os_kernel.submit_task(visual_input, synchronous=True)
        process_time = time.time() - start_time
        
        action = result.get("action")
        if action:
            action_name = action.get('action', 'Unknown')
            confidence = action.get('value', 0.0)
        else:
            action_name = "No Action"
            confidence = 0.0
            
        print(f"   🤖 Action Selected: '{action_name}' (Confidence: {confidence:.2f})")
        
        if brain.use_kernel and brain.kernel_substrate:
            ops = brain.kernel_substrate.kernel.stats['ops']
            spikes = brain.kernel_substrate.kernel.stats['spikes']
            print(f"   ⚡ Kernel Stats: {ops} ops (cumulative), {spikes} spikes processed")
        
        brain.motivation_system.update_state({"reward": 1.0})
        print(f"   ⚡ Processing Time: {process_time*1000:.1f}ms")
        time.sleep(0.5)

    print("\n💾 Capturing Brain Activity History (Before Sleep)...")
    history_buffer = []
    if brain.use_kernel and brain.kernel_substrate:
        # ディープコピーしておく（参照渡しだとresetで消える可能性があるため）
        history_buffer = list(brain.kernel_substrate.kernel.spike_history)
        print(f"   -> Captured {len(history_buffer)} spike events.")

    print("\n🌙 [PHASE 2] SLEEP CYCLE - Memory Consolidation")
    os_kernel.shutdown() # ここで履歴がクリアされる
    time.sleep(1.0)
    os_kernel.boot()
    
    print("\n🎨 [PHASE 3] VISUALIZATION")
    if history_buffer:
        save_raster_plot(history_buffer)
    else:
        print("⚠️ No history captured to plot.")

    print("\n✅ Demo Completed Successfully.")

if __name__ == "__main__":
    run_demo()