# ファイルパス: app/main.py
# 日本語タイトル: DORA Observer Interface (Fix: Kernel -> Substrate)
# 目的・内容:
#   Neuromorphic OSの観測・操作用Webインターフェース。
#   Gradioを使用し、脳の状態（スパイク、意識）を可視化する。
#   v3.2対応: brain.kernelをbrain.substrateに変更。

import logging
import time
from typing import Any, Dict, List, Tuple

import gradio as gr
import torch
from app.containers import AppContainer

# ロギング設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_ui(container: AppContainer) -> gr.Blocks:
    """UIの構築"""
    chat_service = container.chat_service()
    brain = container.brain()

    with gr.Blocks(title="DORA: Neuromorphic Research OS", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🧠 DORA: Neuromorphic Research OS Observer
            
            知能の「機能」ではなく「現象」を観測するためのダッシュボード。
            """
        )

        with gr.Row():
            # 左カラム: チャットとインタラクション
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Cognitive Stream (Consciousness Log)", height=500)
                msg = gr.Textbox(
                    label="Sensory Input (Text)",
                    placeholder="脳への入力メッセージを入力してください...",
                    lines=2,
                )
                with gr.Row():
                    submit_btn = gr.Button("送信 (Inject Input)", variant="primary")
                    clear_btn = gr.Button("リセット")

            # 右カラム: 脳内部状態モニタ
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Internal State Monitor")
                
                with gr.Group():
                    cycle_monitor = gr.Number(label="Total Cycles", value=0)
                    system_status = gr.Textbox(label="System Status", value="BOOTING")
                    phase_monitor = gr.Textbox(label="Current Phase", value="Wake")
                
                with gr.Accordion("Neural Activity (Spikes)", open=True):
                    spikes_monitor = gr.JSON(label="Active Neurons Count")
                
                with gr.Accordion("Global Workspace (Consciousness)", open=False):
                    consciousness_monitor = gr.JSON(label="Broadcast Content")

        def bot_response(message: str, history: List[Tuple[str, str]]) -> Any:
            """
            ユーザー入力に対する応答処理と、脳状態の観測更新。
            """
            if not message:
                return history, 0, "Running", "Wake", {}, {}

            # 1. 外部入力の処理 (ChatService経由)
            # 実際にはここでSNNへのエンコーディングや推論が行われる
            try:
                response = chat_service.chat(message)
            except Exception as e:
                logger.error(f"Chat service error: {e}")
                response = f"Error: {str(e)}"

            # 2. OSサイクルの実行 (擬似的な感覚入力としてランダムノイズを使用)
            # 本来はテキストエンコーダーからのスパイクを入力する
            dummy_sensory_input = torch.randn(1, 784)
            observation = brain.run_cycle(dummy_sensory_input)

            # 3. 状態の取得と整形
            # [修正] brain.kernel -> brain.substrate
            raw_spikes = brain.substrate.prev_spikes
            spike_summary = {}
            
            if raw_spikes:
                for region, tensor in raw_spikes.items():
                    if tensor is not None:
                        # TensorをPythonのintに変換して表示
                        count = int(tensor.sum().item())
                        spike_summary[region] = f"{count} spikes"

            # 意識状態の取得
            consciousness_data = {
                "intensity": float(brain.global_workspace.get_current_thought().mean().item()),
                "content_source": "Thinking..." # 仮
            }

            # 履歴の更新
            history.append((message, response))

            return (
                history,
                observation.get("cycle", 0),
                observation.get("status", "RUNNING"),
                observation.get("phase", "wake"),
                spike_summary,
                consciousness_data
            )

        # イベントハンドラの設定
        submit_btn.click(
            bot_response,
            inputs=[msg, chatbot],
            outputs=[
                chatbot,
                cycle_monitor,
                system_status,
                phase_monitor,
                spikes_monitor,
                consciousness_monitor,
            ],
        )
        
        # テキストボックスでEnterキーを押した時も送信
        msg.submit(
            bot_response,
            inputs=[msg, chatbot],
            outputs=[
                chatbot,
                cycle_monitor,
                system_status,
                phase_monitor,
                spikes_monitor,
                consciousness_monitor,
            ],
        )

        # 入力欄をクリア
        msg.submit(lambda: "", None, msg) 
        submit_btn.click(lambda: "", None, msg)

        # リセットボタン
        def reset_history():
            return [], 0, "RESET", "Wake", {}, {}
            
        clear_btn.click(
            reset_history,
            None,
            [chatbot, cycle_monitor, system_status, phase_monitor, spikes_monitor, consciousness_monitor],
        )

    return demo


def main():
    """アプリケーションのエントリーポイント"""
    logger.info("🔌 Wiring application container...")
    container = AppContainer()
    container.wire(modules=[__name__])

    # 脳の起動
    logger.info("🧠 Booting Neuromorphic OS...")
    brain = container.brain()
    try:
        brain.boot()
    except Exception as e:
        logger.error(f"Failed to boot brain: {e}")

    # UIの作成と起動
    logger.info("🚀 Launching User Interface...")
    demo = create_ui(container)
    
    # 共有リンクが必要な場合は share=True に設定
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()