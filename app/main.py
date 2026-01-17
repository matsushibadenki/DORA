# ファイルパス: app/main.py
# 日本語タイトル: DORA Research Observer Dashboard (Gradio 5.x Fix)
# 目的・内容:
#   Neuromorphic Research OSの状態をリアルタイムで観測するためのダッシュボード。
#   Gradio 5.xの仕様(Messages formatがデフォルトかつtype引数なし)に対応。

import sys
import os

# --- プロジェクトルートをsys.pathに追加 ---
# python app/main.py で実行した場合でもモジュール解決できるようにする
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
# ---------------------------------------------

import logging
import time
from typing import Any, Dict, List, Tuple, Optional, Union

import gradio as gr
import torch
from app.containers import AppContainer

# ロギング設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_ui(container: AppContainer) -> gr.Blocks:
    """
    Observer UIの構築
    """
    chat_service = container.chat_service()
    brain = container.brain()

    with gr.Blocks(title="DORA: Neuromorphic Research OS", theme=gr.themes.Soft()) as demo:
        # ヘッダーエリア
        gr.Markdown(
            """
            # 🔬 DORA: Neuromorphic Research OS Observer
            
            知能の「機能」ではなく、発生する「現象」を観測するための実験コンソール。
            """
        )

        with gr.Row():
            # 左カラム: 入出力実験エリア
            with gr.Column(scale=2):
                gr.Markdown("### 📡 Signal Injection & Conscious Log")
                
                # チャットボットUIを「意識ストリーム」として再定義
                # Gradio 5.xではデフォルトでMessages format(辞書形式)を期待するため、type引数は不要
                chatbot = gr.Chatbot(
                    label="Global Workspace Stream (Broadcast History)", 
                    height=500
                )
                
                with gr.Group():
                    msg = gr.Textbox(
                        label="Sensory Input Injection (Text/Concept)",
                        placeholder="脳へ注入する信号を入力... (例: 'Apple', 'Pain', 'Hello')",
                        lines=1,
                    )
                    with gr.Row():
                        submit_btn = gr.Button("Inject Signal", variant="primary")
                        clear_btn = gr.Button("Reset State")

            # 右カラム: 生体/神経状態モニタエリア
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Bio-Metrics & Substrate")
                
                with gr.Group():
                    cycle_monitor = gr.Number(label="Total Cycles", value=0)
                    status_monitor = gr.Textbox(label="OS Status", value="BOOTING")
                    phase_monitor = gr.Textbox(label="Circadian Phase", value="Wake")
                
                # アコーディオンで詳細情報を表示
                with gr.Accordion("🧠 Neural Activity (Firing Rate)", open=True):
                    spikes_monitor = gr.JSON(label="Region Activity")
                
                with gr.Accordion("🧪 Neuromodulators & Energy", open=True):
                    bio_monitor = gr.JSON(label="Homeostasis")

                with gr.Accordion("🕸️ Connectivity (Synapses)", open=False):
                    synapse_monitor = gr.Number(label="Active Synapses")

        def bot_response(message: str, history: List[Dict[str, str]]) -> Any:
            """
            ユーザー入力に対する応答処理と、脳状態の観測更新。
            修正: historyを辞書形式のリスト [{"role": "user", "content": ...}, ...] として処理
            """
            if history is None:
                history = []

            if not message:
                # 何も入力がない場合でもサイクルは回す（脳は止まらない）
                pass

            # 1. 外部入力処理 (言語野シミュレーションとしてChatServiceを使用)
            response_text = "..."
            try:
                if message:
                    raw_response = chat_service.chat(message)
                    response_text = str(raw_response)
            except Exception as e:
                logger.error(f"Signal processing error: {e}")
                response_text = f"Error: {str(e)}"

            # 2. OSサイクルの実行 
            # (本来はエンコーディングされたスパイク列だが、ここではデモ用にランダムノイズ+入力強度)
            # 入力がある場合、V1への入力強度を高める
            input_intensity = 1.0 if message else 0.1
            dummy_sensory_input = torch.randn(1, 784) * input_intensity
            
            # 脳の1ステップ実行
            observation = brain.run_cycle(dummy_sensory_input)

            # 3. 観測データの整形
            # 神経発火状況
            raw_spikes = observation.get("substrate_activity", {})
            spike_summary = {k: f"{v:.4f} Hz" for k, v in raw_spikes.items()}

            # 生体指標
            bio_data = observation.get("bio_metrics", {})
            
            # 履歴更新 (Messages format / 辞書形式)
            if message:
                history.append({"role": "user", "content": f"[INJECT] {message}"})
                history.append({"role": "assistant", "content": f"[BROADCAST] {response_text}"})
            else:
                # 入力がない場合の自発活動ログ（必要であればここでhistoryに追加）
                pass

            return (
                history,
                observation.get("cycle", 0),
                observation.get("status", "RUNNING"),
                observation.get("phase", "wake"),
                spike_summary,
                bio_data,
                observation.get("synapse_count", 0)
            )

        # イベントハンドラ
        submit_btn.click(
            bot_response,
            inputs=[msg, chatbot],
            outputs=[
                chatbot,
                cycle_monitor,
                status_monitor,
                phase_monitor,
                spikes_monitor,
                bio_monitor,
                synapse_monitor
            ],
        )
        
        msg.submit(
            bot_response,
            inputs=[msg, chatbot],
            outputs=[
                chatbot,
                cycle_monitor,
                status_monitor,
                phase_monitor,
                spikes_monitor,
                bio_monitor,
                synapse_monitor
            ],
        )

        # 入力欄クリア
        msg.submit(lambda: "", None, msg) 
        submit_btn.click(lambda: "", None, msg)

        # リセット処理
        def reset_system():
            logger.info("System Reset Requested.")
            brain.boot() # OS再起動
            return [], 0, "RESET", "Wake", {}, {}, 0
            
        clear_btn.click(
            reset_system,
            None,
            [chatbot, cycle_monitor, status_monitor, phase_monitor, spikes_monitor, bio_monitor, synapse_monitor],
        )

    return demo


def main():
    """アプリケーションエントリーポイント"""
    logger.info("🔌 Wiring application container...")
    container = AppContainer()
    container.wire(modules=[__name__])

    # OS起動プロセス
    logger.info("🧠 Booting Neuromorphic OS...")
    brain = container.brain()
    try:
        brain.boot()
    except Exception as e:
        logger.error(f"Failed to boot brain: {e}")

    # UI起動
    logger.info("🚀 Launching Research Observer...")
    demo = create_ui(container)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()