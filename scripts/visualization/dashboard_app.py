# ファイルパス: scripts/visualization/dashboard_app.py
# 日本語タイトル: Neuromorphic OS Observation Dashboard (Layer 4 GUI)
# 目的・内容:
#   Streamlitを使用したWebベースのデバッグツール。
#   実験結果（metrics.json, system_events.json, heatmaps）を読み込み、
#   脳のエネルギー状態、タスク実行履歴、神経活動を可視化する。

import streamlit as st
import json
import pandas as pd
import plotly.express as px # type: ignore
import plotly.graph_objects as go # type: ignore
import os
import glob
from PIL import Image
import time

# --- Configuration ---
RESULTS_DIR = "benchmarks/results"
st.set_page_config(
    page_title="Neuromorphic OS Dashboard",
    page_icon="🧠",
    layout="wide"
)

# --- Helper Functions ---

@st.cache_data(ttl=5) # 5秒ごとにキャッシュ更新（実行中の実験も追跡可能に）
def load_experiments():
    """結果ディレクトリにある実験フォルダの一覧を取得"""
    if not os.path.exists(RESULTS_DIR):
        return []
    # ディレクトリのみ、かつタイムスタンプ順にソート
    dirs = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]
    dirs.sort(reverse=True) # 最新が上
    return dirs

def load_data(experiment_id):
    """指定された実験のデータを読み込む"""
    base_path = os.path.join(RESULTS_DIR, experiment_id)
    data = {}
    
    # 1. Metrics (Energy, Fatigue etc.)
    metrics_path = os.path.join(base_path, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            data["metrics"] = json.load(f)
    else:
        data["metrics"] = {}

    # 2. System Events (Scheduler Logs)
    events_path = os.path.join(base_path, "system_events.json")
    if os.path.exists(events_path):
        with open(events_path, 'r') as f:
            data["events"] = json.load(f)
    else:
        data["events"] = []

    # 3. Heatmaps
    heatmap_dir = os.path.join(base_path, "plots", "heatmaps")
    data["heatmap_files"] = sorted(glob.glob(os.path.join(heatmap_dir, "*.png")))
    
    return data

# --- UI Components ---

def render_sidebar():
    st.sidebar.title("🧠 Layer 4: Observer")
    st.sidebar.markdown("---")
    
    experiments = load_experiments()
    if not experiments:
        st.sidebar.warning("No experiment results found.")
        st.stop()
        
    selected_exp = st.sidebar.selectbox("Select Experiment", experiments)
    st.sidebar.info(f"ID: {selected_exp}")
    
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        
    return selected_exp

def render_metrics_chart(metrics_data):
    """エネルギーと疲労度のチャートを描画"""
    if not metrics_data:
        st.warning("No metrics data available.")
        return

    # Pandas DataFrameに変換
    df_list = []
    for name, values in metrics_data.items():
        if name in ["energy", "fatigue", "current_energy"]: # 表示したいメトリクス
            for entry in values:
                df_list.append({
                    "step": entry["step"],
                    "value": entry["value"],
                    "metric": name
                })
    
    if not df_list:
        return

    df = pd.DataFrame(df_list)
    
    fig = px.line(df, x="step", y="value", color="metric", 
                  title="Life Signals (Energy & Fatigue)",
                  markers=True,
                  color_discrete_map={"energy": "#00CC96", "fatigue": "#EF553B", "current_energy": "#00CC96"})
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

def render_event_timeline(events):
    """タスク実行とドロップのタイムライン"""
    if not events:
        st.warning("No event data available.")
        return

    # データを整形
    timeline_data = []
    for e in events:
        step = e.get("step", 0)
        event_type = e.get("type", "unknown")
        
        if event_type == "scheduler_step":
            # 実行されたタスク
            for task in e.get("details", {}).get("executed", []):
                timeline_data.append(dict(Step=step, Task=task, Status="Executed", Color="green"))
        
        elif event_type == "task_dropped":
            # ドロップされたタスク
            task_name = e.get("details", {}).get("process", "Unknown")
            reason = e.get("details", {}).get("reason", "")
            timeline_data.append(dict(Step=step, Task=f"{task_name} ({reason})", Status="Dropped", Color="red"))
            
        elif event_type == "phase_change":
            # フェーズ変更
            to_phase = e.get("details", {}).get("to", "")
            timeline_data.append(dict(Step=step, Task=f"Phase -> {to_phase}", Status="PhaseChange", Color="blue"))

    if not timeline_data:
        st.info("No scheduler events logged yet.")
        return

    df = pd.DataFrame(timeline_data)
    
    # 散布図でタイムラインを表現
    fig = px.scatter(df, x="Step", y="Task", color="Status", symbol="Status",
                     title="OS Scheduler Timeline",
                     color_discrete_map={"Executed": "#00CC96", "Dropped": "#EF553B", "PhaseChange": "#636EFA"},
                     size_max=15)
    
    fig.update_traces(marker=dict(size=12))
    fig.update_layout(height=350, yaxis={'categoryorder':'category ascending'})
    st.plotly_chart(fig, use_container_width=True)

def render_brain_viewer(heatmap_files):
    """脳活動ヒートマップのビューア"""
    st.subheader("📸 Brain Activity Viewer")
    
    if not heatmap_files:
        st.info("No heatmap images found.")
        return

    # スライダーで画像選択
    if len(heatmap_files) > 1:
        idx = st.slider("Time Step", 0, len(heatmap_files)-1, 0)
    else:
        idx = 0
        
    image_path = heatmap_files[idx]
    filename = os.path.basename(image_path)
    
    image = Image.open(image_path)
    st.image(image, caption=f"{filename}", use_column_width=False, width=600)

def render_raw_logs(events):
    """生ログの表示"""
    with st.expander("📝 Raw System Logs"):
        st.json(events)

# --- Main App Logic ---

def main():
    experiment_id = render_sidebar()
    data = load_data(experiment_id)
    
    st.title(f"Experiment: {experiment_id}")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("❤️ Homeostasis Monitor")
        render_metrics_chart(data.get("metrics", {}))
        
    with col2:
        render_brain_viewer(data.get("heatmap_files", []))

    st.markdown("---")
    st.subheader("🚦 Scheduler Decisions")
    render_event_timeline(data.get("events", []))
    
    render_raw_logs(data.get("events", []))

if __name__ == "__main__":
    main()