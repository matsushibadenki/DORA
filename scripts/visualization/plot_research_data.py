# ファイルパス: scripts/visualization/plot_research_data.py
# 日本語タイトル: Research Data Visualizer
# 目的・内容:
#   experiment_history.json を読み込み、
#   バイオメトリクス、神経活動、シナプス数の時系列変化をプロットする。

import json
import matplotlib.pyplot as plt
import os
import sys

def plot_results(file_path="runtime_state/experiment_history.json"):
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return

    with open(file_path, "r") as f:
        data = json.load(f)

    # データ展開
    cycles = [d["cycle"] for d in data]
    phases = [d["phase"] for d in data]
    energy = [d["energy"] for d in data]
    fatigue = [d["fatigue"] for d in data]
    synapses = [d["synapse_count"] for d in data]
    consciousness = [d["consciousness"] for d in data]
    memory = [d["memory_stored"] for d in data]
    
    # プロット設定
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    # 1. Phase & Homeostasis (Sleep/Wake & Energy)
    ax1 = axs[0]
    ax1.set_title("Homeostasis & Circadian Rhythm")
    ax1.plot(cycles, energy, label="Energy", color="orange")
    ax1.plot(cycles, fatigue, label="Fatigue", color="gray", linestyle="--")
    # 背景色でフェーズを表現 (Sleep=Blue tint)
    ax1.fill_between(cycles, 0, 1000, where=[p==0 for p in phases], color='blue', alpha=0.1, label="Sleep Phase")
    ax1.set_ylabel("Level")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # 2. Neural Activity & Consciousness
    ax2 = axs[1]
    ax2.set_title("Consciousness & Neural Activity")
    act_assoc = [d["act_assoc"] for d in data]
    ax2.plot(cycles, consciousness, label="Consciousness Level", color="purple", linewidth=2)
    ax2.plot(cycles, act_assoc, label="Assoc. Activity", color="green", alpha=0.5)
    ax2.set_ylabel("Activity / Level")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # 3. Memory & Learning
    ax3 = axs[2]
    ax3.set_title("Hippocampal Memory Buffer")
    ax3.plot(cycles, memory, label="Stored Episodes", color="blue")
    ax3.set_ylabel("Count")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    # 4. Structural Plasticity (Synapse Count)
    ax4 = axs[3]
    ax4.set_title("Structural Plasticity (Active Synapses)")
    ax4.plot(cycles, synapses, label="Synapse Count", color="red")
    ax4.set_ylabel("Count")
    ax4.set_xlabel("Simulation Cycle")
    ax4.legend(loc="upper right")
    ax4.grid(True, alpha=0.3)

    # 保存
    save_path = "runtime_state/research_plot.png"
    plt.savefig(save_path)
    print(f"📊 Plot saved to {save_path}")
    # plt.show() # サーバー環境等でエラーになる場合はコメントアウト

if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else "runtime_state/experiment_history.json"
    plot_results(target_file)