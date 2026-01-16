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
                "timestamp": data.get("timestamp", 0)
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
                "total_activity": 0.0
            }

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

        # 3. ステータス表示
        # 最終更新からの経過時間を計算
        last_update = time.time() - state.get("timestamp", 0)
        connection_status = "🟢 Online" if last_update < 5.0 else "🔴 Offline / Idle"
        
        status_text = (
            f"### 🖥️ Observer Status: {connection_status}\n"
            f"**System Status:** {state['status']}\n"
            f"**Cycle:** {state['cycle']}\n"
            f"**Phase:** {state['phase']}\n"
            f"**Brain Activity:** {state['total_activity']:.4f}\n"
        )

        return df_metabolism, df_activity, status_text

    def launch(self, share: bool = False):
        """ダッシュボードの起動"""
        with gr.Blocks(title="Neuromorphic OS Dashboard") as demo:
            gr.Markdown("# 🧠 Neuromorphic OS - Realtime Observer")
            gr.Markdown("共有ステートファイル (`runtime_state/brain_activity.json`) を監視中...")
            
            with gr.Row():
                with gr.Column(scale=1):
                    status_display = gr.Markdown("Waiting for signal...")
                    refresh_btn = gr.Button("Manual Refresh")
                
                with gr.Column(scale=2):
                    # 代謝グラフ
                    metabolism_plot = gr.LinePlot(
                        x="Cycle",
                        y="Energy",
                        title="Metabolism Dynamics",
                        tooltip=["Cycle", "Energy", "Fatigue"]
                    )
            
            with gr.Row():
                # 領域別活動グラフ
                activity_plot = gr.BarPlot(
                    x="Region",
                    y="Firing Rate",
                    title="Regional Neural Activity",
                    tooltip=["Region", "Firing Rate"],
                    y_lim=[0, 1.0]
                )

            # --- 修正箇所: Timerを使用 ---
            timer = gr.Timer(value=1.0) # 1秒ごとにイベント発火

            # 定期更新イベント (Timer)
            timer.tick(
                fn=self.update_charts,
                inputs=[],
                outputs=[metabolism_plot, activity_plot, status_display]
            )

            # 初期ロード時にも実行
            demo.load(
                fn=self.update_charts,
                inputs=[],
                outputs=[metabolism_plot, activity_plot, status_display]
            )
            
            # 手動リフレッシュ
            refresh_btn.click(
                fn=self.update_charts,
                inputs=[],
                outputs=[metabolism_plot, activity_plot, status_display]
            )

        print("📊 Launching File-based Dashboard...")
        demo.launch(share=share, server_port=7861)

if __name__ == "__main__":
    dashboard = BrainDashboard()
    dashboard.launch()