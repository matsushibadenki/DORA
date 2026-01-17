# ファイルパス: scripts/visualization/plot_memory_learning.py
import json
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve():
    # データ読み込み
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../../runtime_state/memory_experiment_history.json")
    
    if not os.path.exists(data_path):
        print("❌ No data found.")
        return

    with open(data_path, "r") as f:
        history = json.load(f)

    cycles = [h["cycle"] for h in history]
    accuracy = [h["accuracy"] for h in history]
    energy = [h["energy"] for h in history]
    dopamine = [h.get("dopamine", 0) for h in history]
    synapses = [h.get("synapse_count", 0) for h in history]
    phases = [h["phase"] for h in history] 

    # グラフ作成 (3段)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # 1. Learning Curve & Dopamine
    ax1.set_title("Learning & Reinforcement")
    ax1.plot(cycles, accuracy, color="tab:green", linewidth=2, label="Accuracy")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(cycles, dopamine, color="tab:purple", linestyle="--", alpha=0.6, label="Dopamine")
    ax1_twin.set_ylabel("Dopamine Level")
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    # 2. Structural Plasticity (Synapse Count)
    ax2.set_title("Structural Plasticity (Brain Connectivity)")
    ax2.plot(cycles, synapses, color="tab:orange", linewidth=2, label="Active Synapses")
    ax2.set_ylabel("Synapse Count")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    # 3. Energy Cycle
    ax3.set_title("Metabolic State")
    ax3.plot(cycles, energy, color="tab:blue", label="Energy")
    ax3.set_ylabel("Energy Level")
    ax3.set_xlabel("Cycle")
    ax3.grid(True, alpha=0.3)

    # 背景色の塗り分け (Sleep)
    for i in range(len(cycles)-1):
        if phases[i] == "sleep":
            for ax in [ax1, ax2, ax3]:
                ax.axvspan(cycles[i], cycles[i+1], color='gray', alpha=0.2, lw=0)

    # 保存
    out_file = "memory_learning_result.png"
    plt.savefig(out_file, dpi=150)
    print(f"✅ Graph saved to {out_file}")

if __name__ == "__main__":
    plot_learning_curve()