# ファイルパス: app/main.py
# 日本語タイトル: Neuromorphic OS UI (with Brain Monitor)
# 修正内容:
#   - Gradio UIに画像出力コンポーネント(brain_monitor)を追加。
#   - ChatServiceからの応答に含まれる統計情報を使って画像を更新するロジックを追加。

import gradio as gr
import argparse
import logging
import sys
import os
import traceback
import yaml
import numpy as np

from app.containers import AppContainer
# プロッターのインポート
from snn_research.visualization.spike_plotter import SpikePlotter

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_ui(container: AppContainer) -> gr.Blocks:
    """Gradio UIの構築"""
    
    # サービス取得 (SNNエンジンへのアクセスが必要なため、containerからOSを取得)
    # ChatService経由ではなく、UI側で描画するためにOSインスタンスも参照
    brain = container.neuromorphic_os()
    chat_service = container.chat_service()

    with gr.Blocks(title="Neuromorphic OS Dashboard", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧠 Neuromorphic Research OS v1.0")
        
        with gr.Row():
            # 左カラム: チャット
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(label="Consciousness Stream", height=400)
                msg = gr.Textbox(show_label=False, placeholder="Talk to the brain...", scale=4)
                with gr.Row():
                    submit_btn = gr.Button("Send Input", variant="primary")
                    clear_btn = gr.Button("Reset State")

            # 右カラム: モニター
            with gr.Column(scale=1):
                with gr.Tab("Brain Activity"):
                    # 脳活動を表示する画像エリア
                    brain_monitor = gr.Image(
                        label="Cortical Activity (V1 | Assoc | Motor)", 
                        type="numpy",
                        interactive=False
                    )
                    stats_box = gr.Markdown("### Status: Waiting for stimuli...")

        # --- イベントハンドラ ---

        def user_message(user_input, history):
            if history is None: history = []
            return "", history + [[user_input, None]]

        def bot_response(history):
            if not history: return history, "", None

            user_input = history[-1][0]
            past_history = history[:-1]
            
            # ChatServiceからストリーム応答を取得
            stream_gen = chat_service.stream_response(user_input, past_history)
            
            try:
                for updated_history, stats in stream_gen:
                    # 最新の脳状態を取得して画像化
                    # (本来はstream_genがstateも返すべきだが、今回はOSから直接現在の状態を覗き見る)
                    # ※並列処理ではないため、この瞬間の状態を取得可能
                    
                    # 最後の forward_step で保存された prev_spikes を可視化
                    current_state = {"spikes": brain.kernel.prev_spikes}
                    brain_img = SpikePlotter.plot_substrate_state(current_state)
                    
                    # 統計テキスト作成
                    if isinstance(stats, dict):
                        stats_text = f"""
                        **Cycle:** {stats.get('step', 0)}
                        **Total Spikes:** {stats.get('total_spikes', 0)}
                        **Motor Output:** {stats.get('last_motor', '')}
                        """
                    else:
                        stats_text = str(stats)

                    yield updated_history, stats_text, brain_img
                    
            except Exception as e:
                logger.error(f"Error: {e}")
                traceback.print_exc()
                history[-1][1] = f"Error: {str(e)}"
                yield history, "Error", None

        # --- イベント連携 ---
        msg.submit(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_response, [chatbot], [chatbot, stats_box, brain_monitor]
        )
        
        submit_btn.click(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_response, [chatbot], [chatbot, stats_box, brain_monitor]
        )
        
        clear_btn.click(lambda: [], None, chatbot, queue=False)

    return demo

def main():
    parser = argparse.ArgumentParser(description="Neuromorphic OS Interface")
    parser.add_argument("--config", type=str, default="configs/templates/base_config.yaml", help="Path to config file")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    args = parser.parse_args()

    container = AppContainer()
    
    if os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config_data = yaml.safe_load(f)
            container.config.from_dict(config_data)
        except Exception:
            pass
    
    container.wire(modules=[__name__])
    
    try:
        os_system = container.neuromorphic_os()
        os_system.boot()
    except Exception as e:
        logger.critical(f"Boot Failed: {e}")
        return

    logger.info("Constructing UI with Visualization...")
    demo = create_ui(container)
    demo.queue().launch(server_name=args.host, server_port=args.port, share=False)

if __name__ == "__main__":
    main()