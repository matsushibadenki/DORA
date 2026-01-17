# ファイルパス: scripts/visualization/plot_research_data.py
# 日本語タイトル: Research Data Visualizer (Fixed for JSON Keys)
# 目的・内容:
#   実験結果(experiment_history.json)を読み込み、グラフ化する。
#   修正: アップロードされたJSONのキー(fatigue_level)に完全対応。

import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

def plot_experiment_data():
    # 1. データファイルのパス設定
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # プロジェクトルート/runtime_state/experiment_history.json を参照
    data_path = os.path.join(current_dir, "../../runtime_state/experiment_history.json")
    data_path = os.path.abspath(data_path)

    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("   Please run 'python scripts/experiments/run_research_cycle.py' first.")
        return

    print(f"📂 Loading data from: {data_path}")
    try:
        with open(data_path, "r") as f:
            history = json.load(f)
    except json.JSONDecodeError:
        print("❌ Failed to parse JSON. File might be corrupted or empty.")
        return

    print(f"📊 Analyzing {len(history)} cycles of data...")

    # 2. データの抽出（JSON構造に合わせてキーを指定）
    cycles = [h["cycle"] for h in history]
    
    # Bio Metrics
    energy = [h["bio_metrics"].get("current_energy", 0) for h in history]
    
    # キー名の揺らぎに対応 (fatigue_level / fatigue_index / fatigue)
    fatigue = []
    for h in history:
        bio = h["bio_metrics"]
        val = bio.get("fatigue_level") or bio.get("fatigue_index") or bio.get("fatigue") or 0
        fatigue.append(val)

    consciousness = [h.get("consciousness_level", 0) for h in history]
    
    # 神経活動（Association野）
    assoc_activity = [h["substrate_activity"].get("Association", 0) for h in history]
    
    # フェーズ情報 (Wake=1, Sleep=0)
    phases = [1 if h["phase"] == "wake" else 0 for h in history]

    # 3. プロット作成
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    # --- Graph 1: Metabolic State (Energy vs Fatigue) ---
    color_energy = "tab:blue"
    color_fatigue = "tab:red"
    
    ax1.set_title("Metabolic State (Homeostasis)", fontsize=14)
    ax1.plot(cycles, energy, label="Energy", color=color_energy, linewidth=2)
    ax1.set_ylabel("Energy Level", color=color_energy, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color_energy)
    ax1.grid(True, alpha=0.3)
    
    # 疲労度を右軸に
    ax1_twin = ax1.twinx()
    ax1_twin.plot(cycles, fatigue, label="Fatigue", color=color_fatigue, linestyle="--", alpha=0.8)
    ax1_twin.set_ylabel("Fatigue Level", color=color_fatigue, fontsize=12)
    ax1_twin.tick_params(axis='y', labelcolor=color_fatigue)
    
    # --- Graph 2: Neural Activity (Firing Rate) ---
    ax2.set_title("Neural Activity (Association Cortex)", fontsize=14)
    ax2.plot(cycles, assoc_activity, label="Association Cortex", color="tab:orange", linewidth=1.5)
    ax2.set_ylabel("Mean Firing Rate", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    # --- Graph 3: Consciousness & Phase ---
    ax3.set_title("Consciousness Level & Wake/Sleep Phase", fontsize=14)
    ax3.plot(cycles, consciousness, label="Global Workspace (Consciousness)", color="tab:purple", linewidth=2)
    ax3.set_ylabel("Salience / Awareness", fontsize=12)
    ax3.set_xlabel("Cycle Time", fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 意識レベルの領域塗りつぶし
    ax3.fill_between(cycles, consciousness, color="tab:purple", alpha=0.2)

    # 4. 背景色によるフェーズ表現 (Wake=白, Sleep=グレー)
    # 変化点を検出して塗り分ける
    for i in range(len(cycles) - 1):
        if phases[i] == 0: # Sleep phase
            for ax in [ax1, ax2, ax3]:
                ax.axvspan(cycles[i], cycles[i+1], color='gray', alpha=0.15, lw=0)

    # 保存
    output_file = "experiment_result.png"
    print("🎨 Generating plot...")
    plt.savefig(output_file, dpi=150)
    print(f"✅ Plot saved to '{output_file}'")
    
    # 環境によってはウィンドウ表示
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    plot_experiment_data()