# ファイルパス: app/dashboard.py
# 日本語タイトル: Neuromorphic OS Dashboard (Timer Fix)
# 目的・内容:
#   Neuromorphic OSの内部状態を可視化するダッシュボード。
#   修正: Gradio 4.20.0対応のため、demo.loadのevery引数を廃止し、
#   gr.Timerコンポーネントを使用して定期更新を実装。

import json
import logging
import os
import time
from typing import Dict, Any, List

import gradio as gr
import pandas as pd
import psutil
import subprocess
import threading

logger = logging.getLogger(__name__)


class BrainDashboard:
    """
    ファイルベースの脳活動オブザーバー。
    runtime_state/brain_activity.json を定期的に読み取り、可視化する。
    """

    def __init__(self):
        self.state_file_path = "runtime_state/brain_activity.json"

        # ログデータのバッファ
        self.history_energy: List[float] = []
        self.history_fatigue: List[float] = []
        self.history_cycles: List[int] = []
        self.max_history = 100

    def read_brain_state(self) -> Dict[str, Any]:
        """
        JSONファイルから最新の脳状態を読み取る。
        """
        if not os.path.exists(self.state_file_path):
            return {
                "status": "WAITING_FOR_KERNEL",
                "cycle": 0,
                "phase": "Connecting...",
                "energy": 0.0,
                "fatigue": 0.0,
                "substrate_activity": {},
                "total_activity": 0.0
            }

        try:
            with open(self.state_file_path, "r") as f:
                data = json.load(f)

            # データの整形
            activity = data.get("substrate_activity", {})
            total_activity = sum(activity.values())

            return {
                "status": data.get("status", "UNKNOWN"),
                "cycle": data.get("cycle", 0),
                "phase": data.get("phase", "Unknown"),
                "energy": data.get("energy", 0.0),
                "fatigue": data.get("fatigue", 0.0),
                "substrate_activity": activity,
                "total_activity": total_activity,
                "timestamp": data.get("timestamp", 0),
                # System Metrics (Real-time)
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
            }
        except Exception as e:
            logger.error(f"Error reading state file: {e}")
            return {
                "status": "READ_ERROR",
                "cycle": 0,
                "phase": "Error",
                "energy": 0.0,
                "fatigue": 0.0,
                "substrate_activity": {},
                "total_activity": 0.0,
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
            }

    def determine_health_status(self, state: Dict[str, Any]) -> str:
        """
        現在の状態からヘルスステータスを判定
        Returns: "Healthy" | "Warning" | "Critical"
        """
        if state["status"] == "READ_ERROR":
            return "Critical"

        # 1. Latency Check
        last_update = time.time() - state.get("timestamp", 0)
        if last_update > 30.0:
            return "Critical"  # 30秒以上更新なし
        elif last_update > 10.0:
            return "Warning"

        # 2. System Resource Check
        if state["cpu_percent"] > 90.0 or state["memory_percent"] > 90.0:
            return "Warning"

        # 3. Bio-Integrity Check
        # fatigueがenergyを超過しそうな場合など（簡易判定）
        energy = state.get("energy", 0.0)
        fatigue = state.get("fatigue", 0.0)
        if energy < 100.0 and fatigue > 500.0:
            return "Warning"

        return "Healthy"

    def update_charts(self):
        """Gradioの定期更新用関数"""
        state = self.read_brain_state()

        # 履歴の更新
        self.history_cycles.append(state["cycle"])
        self.history_energy.append(state["energy"])
        self.history_fatigue.append(state["fatigue"])

        # バッファ制限
        if len(self.history_cycles) > self.max_history:
            self.history_cycles.pop(0)
            self.history_energy.pop(0)
            self.history_fatigue.pop(0)

        # 1. 代謝グラフ (Metabolism)
        df_metabolism = pd.DataFrame({
            "Cycle": self.history_cycles,
            "Energy": self.history_energy,
            "Fatigue": self.history_fatigue
        })

        # 2. 領域別活性度 (Activity)
        activity_data = state["substrate_activity"]
        if not activity_data:
            activity_data = {"None": 0.0}

        df_activity = pd.DataFrame({
            "Region": list(activity_data.keys()),
            "Firing Rate": list(activity_data.values())
        })

        # 3. ステータス表示 & ヘルスチェック
        # 最終更新からの経過時間
        last_update_diff = time.time() - state.get("timestamp", 0)
        health = self.determine_health_status(state)

        # アイコン決定
        health_icon = "🟢" if health == "Healthy" else "🟡" if health == "Warning" else "🔴"
        connection_status = "Online" if last_update_diff < 5.0 else "Offline / Idle"

        status_text = (
            f"### {health_icon} System Health: {health}\n"
            f"**Connection:** {connection_status} ({last_update_diff:.1f}s ago)\n"
            f"**Cycle:** {state['cycle']} | **Phase:** {state['phase']}\n"
            f"**Brain Status:** {state['status']}\n"
            f"---\n"
            f"**CPU:** {state['cpu_percent']}% | **RAM:** {state['memory_percent']}%\n"
            f"**Energy:** {state['energy']:.1f} | **Fatigue:** {state['fatigue']:.1f}"
        )

        # 4. 視覚野データ (Visual Cortex)
        visual_data = state.get("visual_cortex", {})
        input_img = visual_data.get("input_image", [])
        recon_img = visual_data.get("reconstructed_image", [])

        # Convert to 28x28 numpy array if flat list
        # gr.Image expects numpy array
        import numpy as np

        def to_img_array(data):
            # 784次元のデータがあれば28x28にreshape
            if not data or len(data) != 784:
                return np.zeros((28, 28))
            return np.array(data).reshape(28, 28)

        img_in = to_img_array(input_img)
        img_rec = to_img_array(recon_img)

        # 5. ベンチマーク進捗 (Learning Lab)
        bench_progress = self.read_benchmark_progress()
        df_bench = pd.DataFrame(bench_progress) if bench_progress else pd.DataFrame(
            {"Epoch": [], "Accuracy": []})

        return df_metabolism, df_activity, status_text, img_in, img_rec, df_bench

    def read_benchmark_progress(self):
        """ベンチマーク進捗ファイルの読み込み"""
        path = "runtime_state/benchmark_progress.json"
        if not os.path.exists(path):
            return []
        try:
            # 履歴を蓄積する仕組みがないため、単一の進捗を表示するか、
            # benchmark script側で追記型にする必要があるが、
            # 今回は簡易的に「現在の進捗」を表示する。
            # しかしLinePlotには履歴が必要。
            # ここでは簡易的にdashboard側で履歴を持つか、
            # benchmark scriptが履歴配列を吐くのがベストだが、
            # Dashboardのメモリで履歴を持つことにする。
            with open(path, "r") as f:
                data = json.load(f)

            # data is single dict: {"epoch": 1, "accuracy": 90.0...}
            # We need to append to history
            if not hasattr(self, "bench_history"):
                self.bench_history = []

            # Check if this is a new update
            last_epoch = self.bench_history[-1]["Epoch"] if self.bench_history else -1
            if data["epoch"] != last_epoch:
                self.bench_history.append(
                    {"Epoch": data["epoch"], "Accuracy": data["accuracy"]})

            return self.bench_history
        except:
            return []

    def run_benchmark(self, runs, epochs, threshold):
        """ベンチマークをバックグラウンドで実行"""
        cmd = [
            "python", "benchmarks/stability_benchmark_v2.py",
            "--runs", str(int(runs)),
            "--epochs", str(int(epochs)),
            "--threshold", str(float(threshold))
        ]
        # Reset history
        self.bench_history = []
        # Run in subprocess
        subprocess.Popen(cmd)
        return "🚀 Benchmark Started! Monitoring progress..."

    def launch(self, share: bool = False):
        """ダッシュボードの起動"""
        with gr.Blocks(title="Neuromorphic OS Dashboard") as demo:
            gr.Markdown("# 🧠 Neuromorphic OS - Realtime Observer")
            gr.Markdown(
                "共有ステートファイル (`runtime_state/brain_activity.json`) を監視中...")

            with gr.Row():
                with gr.Column(scale=1):
                    status_display = gr.Markdown("Waiting for signal...")
                    refresh_btn = gr.Button("Manual Refresh")

                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.Tab("Overview"):
                            # 代謝グラフ
                            metabolism_plot = gr.LinePlot(
                                x="Cycle",
                                y="Energy",
                                title="Metabolism Dynamics",
                                tooltip=["Cycle", "Energy", "Fatigue"]
                            )
                            # 領域別活動グラフ
                            activity_plot = gr.BarPlot(
                                x="Region",
                                y="Firing Rate",
                                title="Regional Neural Activity",
                                tooltip=["Region", "Firing Rate"],
                                y_lim=[0, 1.0]
                            )

                        with gr.Tab("👁️ Visual Cortex"):
                            gr.Markdown("### Internal Visual Representation")
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("**Retinal Input (V1)**")
                                    heatmap_in = gr.Image(
                                        show_label=False,
                                        label="Sensory Input",
                                        height=290,
                                        width=290
                                    )
                                with gr.Column():
                                    gr.Markdown("**Top-down Prediction**")
                                    heatmap_rec = gr.Image(
                                        show_label=False,
                                        label="Reconstruction",
                                        height=290,
                                        width=290
                                    )

                        with gr.Tab("🧪 Learning Lab"):
                            gr.Markdown("### Stability Benchmark Runner")
                            with gr.Row():
                                b_runs = gr.Number(
                                    value=1, label="Runs", precision=0)
                                b_epochs = gr.Number(
                                    value=3, label="Epochs/Run", precision=0)
                                b_thresh = gr.Number(
                                    value=80, label="Success Threshold %")
                            b_btn = gr.Button("🚀 Start Benchmark")
                            b_status = gr.Markdown("Ready.")

                            bench_plot = gr.LinePlot(
                                x="Epoch",
                                y="Accuracy",
                                title="Learning Curve",
                                tooltip=["Epoch", "Accuracy"]
                            )

                            b_btn.click(
                                fn=self.run_benchmark,
                                inputs=[b_runs, b_epochs, b_thresh],
                                outputs=[b_status]
                            )

            # --- 修正箇所: Timerを使用 ---
            timer = gr.Timer(value=1.0)  # 1秒ごとにイベント発火

            # 定期更新イベント (Timer)
            timer.tick(
                fn=self.update_charts,
                inputs=[],
                outputs=[metabolism_plot, activity_plot,
                         status_display, heatmap_in, heatmap_rec, bench_plot]
            )

            # 初期ロード時にも実行
            demo.load(
                fn=self.update_charts,
                inputs=[],
                outputs=[metabolism_plot, activity_plot,
                         status_display, heatmap_in, heatmap_rec, bench_plot]
            )

            # 手動リフレッシュ
            refresh_btn.click(
                fn=self.update_charts,
                inputs=[],
                outputs=[metabolism_plot, activity_plot,
                         status_display, heatmap_in, heatmap_rec, bench_plot]
            )

        print("📊 Launching File-based Dashboard...")
        demo.launch(share=share)


if __name__ == "__main__":
    dashboard = BrainDashboard()
    dashboard.launch()
