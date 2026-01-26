# ディレクトリ: app/main.py
# ファイル: DORA Research Observer Dashboard
# 目的: Neuromorphic Research OSの状態をリアルタイムで観測するためのメインダッシュボード。
#       Gradio 5.x/6.xに対応し、BrainからのTensorデータを安全に可視化する。

import sys
import os
import time
import logging
import json
import torch
import gradio as gr
from typing import Any, Dict, List, Union

# --- プロジェクトルートをsys.pathに追加 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
# ---------------------------------------------

from app.containers import AppContainer

# ロギング設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def deep_safe_convert(data: Any) -> Any:
    """
    Brainから出力される複雑なデータ（Tensor, Numpy等）を
    Gradioが確実に表示できるPython標準型（dict, list, int, float, str）に再帰的に変換する。
    """
    if isinstance(data, dict):
        return {str(k): deep_safe_convert(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [deep_safe_convert(v) for v in data]
    elif isinstance(data, (torch.Tensor,)):
        try:
            # スカラーの場合
            return data.item()
        except Exception:
            # 配列の場合
            return [deep_safe_convert(x) for x in data.tolist()]
    elif hasattr(data, 'item'):  # Numpy types
        return data.item()
    elif isinstance(data, (float, int, str, bool, type(None))):
        return data
    else:
        # 変換不能なオブジェクトは文字列化して安全を確保
        return str(data)


def create_ui(container: AppContainer) -> gr.Blocks:
    """Observer UIの構築"""
    chat_service = container.chat_service()
    brain = container.brain()

    with gr.Blocks(title="DORA: Neuromorphic Research OS", theme=gr.themes.Soft()) as demo:
        # ヘッダー
        gr.Markdown(
            """
            # 🔬 DORA: Neuromorphic Research OS Observer
            知能の「機能」ではなく、発生する「現象」を観測するための実験コンソール。
            """
        )

        with gr.Row():
            # --- 左カラム: 対話と入力 ---
            with gr.Column(scale=2):
                gr.Markdown("### 📡 Signal Injection & Conscious Stream")
                
                # Chatbot (Gradio 5.x/6.x対応: type引数なし)
                chatbot = gr.Chatbot(
                    label="Global Workspace Stream",
                    height=500,
                    show_label=True
                )
                
                with gr.Group():
                    msg = gr.Textbox(
                        label="Sensory Input",
                        placeholder="Type a message (e.g. 'hello', 'pain', 'apple')...",
                        lines=1,
                    )
                    with gr.Row():
                        submit_btn = gr.Button("Inject Signal", variant="primary")
                        clear_btn = gr.Button("Reset Brain State")

            # --- 右カラム: 状態モニタリング ---
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Bio-Metrics & Substrate")

                with gr.Group():
                    cycle_monitor = gr.Number(label="Total Cycles", value=0)
                    with gr.Row():
                        status_monitor = gr.Textbox(label="OS Status", value="BOOTING")
                        phase_monitor = gr.Textbox(label="Phase", value="Wake")

                # 安全化されたデータであれば gr.JSON を使用してもフリーズしない
                with gr.Accordion("🧠 Neural Activity (Firing Rate)", open=True):
                    spikes_monitor = gr.JSON(label="Region Activity")

                with gr.Accordion("🧪 Neuromodulators & Energy", open=True):
                    bio_monitor = gr.JSON(label="Homeostasis")

                with gr.Accordion("🕸️ Connectivity", open=False):
                    synapse_monitor = gr.Number(label="Active Synapses")

        def bot_response(message: str, history: List[Any]) -> Any:
            """
            ユーザー入力に対する応答処理と、脳状態の観測更新。
            """
            # 1. 履歴の正規化（Gradioのバージョン差異を吸収）
            new_history = []
            if history:
                for item in history:
                    if isinstance(item, dict):
                        new_history.append(item)
                    elif isinstance(item, (list, tuple)) and len(item) == 2:
                        # 旧形式互換
                        new_history.append({"role": "user", "content": str(item[0])})
                        new_history.append({"role": "assistant", "content": str(item[1])})

            # 2. 処理の実行
            response_text = "..."
            observation = {}
            status_txt = "RUNNING"
            
            try:
                if message:
                    # 会話エンジンの実行
                    raw_res = chat_service.chat(message)
                    response_text = str(raw_res)

                    # 脳シミュレーションの実行 (1サイクル)
                    # ※実稼働時は適切な入力エンコーディングを行うが、ここではデモ用入力
                    dummy_input = torch.randn(1, 784)
                    observation = brain.run_cycle(dummy_input)
                    
            except Exception as e:
                logger.error(f"Execution Error: {e}")
                response_text = f"⚠️ SYSTEM ERROR: {e}"
                status_txt = "ERROR"
                observation = {}

            # 3. データの安全な変換 (ここが重要)
            # Tensor等が含まれる辞書を、JSON化可能な形式に変換する
            safe_observation = deep_safe_convert(observation)
            
            # 各モニタ用データの抽出
            cycle_val = safe_observation.get("cycle", 0)
            status_txt = str(safe_observation.get("status", status_txt))
            phase_txt = str(safe_observation.get("phase", "Wake"))
            
            spikes_data = safe_observation.get("substrate_activity", {})
            bio_data = safe_observation.get("bio_metrics", {})
            synapse_val = safe_observation.get("synapse_count", 0)

            # 4. 履歴の更新
            if message:
                new_history.append({"role": "user", "content": message})
                new_history.append({"role": "assistant", "content": response_text})

            return (
                new_history,
                cycle_val,
                status_txt,
                phase_txt,
                spikes_data,  # 安全な辞書データなので gr.JSON で表示可能
                bio_data,     # 安全な辞書データなので gr.JSON で表示可能
                synapse_val
            )

        # イベントハンドラの設定
        ui_outputs = [
            chatbot,
            cycle_monitor,
            status_monitor,
            phase_monitor,
            spikes_monitor,
            bio_monitor,
            synapse_monitor
        ]

        submit_btn.click(
            bot_response,
            inputs=[msg, chatbot],
            outputs=ui_outputs,
        )
        msg.submit(
            bot_response,
            inputs=[msg, chatbot],
            outputs=ui_outputs,
        )

        # 入力欄の自動クリア
        msg.submit(lambda: "", None, msg)
        submit_btn.click(lambda: "", None, msg)

        # リセット処理
        def reset_system():
            logger.info("System Reset Requested.")
            try:
                brain.boot()
            except Exception as e:
                logger.error(f"Reset failed: {e}")
            # 初期状態を返す
            return [], 0, "RESET", "Wake", {}, {}, 0

        clear_btn.click(
            reset_system,
            None,
            ui_outputs,
        )

    return demo


def main():
    """アプリケーションエントリーポイント"""
    logger.info("🔌 Wiring application container...")
    container = AppContainer()
    container.wire(modules=[__name__])

    logger.info("🧠 Booting Neuromorphic OS...")
    brain = container.brain()
    try:
        brain.boot()
    except Exception as e:
        logger.error(f"Failed to boot brain: {e}")

    logger.info("🚀 Launching Research Observer...")
    demo = create_ui(container)
    
    # 外部公開設定などはここで調整
    demo.queue().launch(
        server_name="127.0.0.1",
        share=False,
        debug=True  # 開発中はTrue推奨
    )


if __name__ == "__main__":
    main()