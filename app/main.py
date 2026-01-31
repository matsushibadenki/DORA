# ファイルパス: app/main.py
# 日本語タイトル: DORA Observer Dashboard (Gradio 6.0 Compatible)
# 目的・内容:
#   Neuromorphic Research OSの状態をリアルタイムで観測するためのWebインターフェース。
#   Gradio 6.0 互換モードで動作するように修正。

import logging
from typing import Any, Dict, List

import gradio as gr
import torch

from app.containers import AppContainer

# ロギング設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def deep_safe_convert(data: Any) -> Any:
    """
    Brainから出力される複雑なデータをGradioが表示可能な型に変換する。
    """
    if isinstance(data, dict):
        return {str(k): deep_safe_convert(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [deep_safe_convert(v) for v in data]
    elif isinstance(data, tuple):
        return [deep_safe_convert(v) for v in data]
    elif isinstance(data, torch.Tensor):
        try:
            if data.numel() == 1:
                return data.item()
            return [deep_safe_convert(x) for x in data.tolist()]
        except Exception:
            return str(data)
    elif hasattr(data, "item"):  # Numpy types
        return data.item()
    elif isinstance(data, (float, int, str, bool, type(None))):
        return data
    else:
        return str(data)


def create_ui(container: AppContainer) -> gr.Blocks:
    """
    Observer UIの構築関数。
    """
    chat_service = container.chat_service()
    brain = container.brain()

    with gr.Blocks(title="DORA: Neuromorphic Research OS") as demo:
        # --- Header ---
        gr.Markdown(
            """
            # 🔬 DORA: Neuromorphic Research OS Observer
            知能の「機能」ではなく、発生する「現象」を観測するための実験コンソール。
            """
        )

        with gr.Row():
            # --- Left Column: Interaction ---
            with gr.Column(scale=2):
                gr.Markdown("### 📡 Signal Injection & Conscious Stream")

                # Gradio 6.0 互換: メッセージ形式(role/content辞書)を使用
                chatbot = gr.Chatbot(
                    label="Global Workspace Stream", height=500, show_label=True
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

            # --- Right Column: Observation ---
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Bio-Metrics & Substrate")

                with gr.Group():
                    cycle_monitor = gr.Number(label="Total Cycles", value=0)
                    with gr.Row():
                        status_monitor = gr.Textbox(label="OS Status", value="BOOTING")
                        phase_monitor = gr.Textbox(label="Phase", value="Wake")

                with gr.Accordion("🧠 Neural Activity (Firing Rate)", open=True):
                    spikes_monitor = gr.JSON(label="Region Activity")

                with gr.Accordion("🧪 Neuromodulators & Energy", open=True):
                    bio_monitor = gr.JSON(label="Homeostasis")

                with gr.Accordion("🕸️ Connectivity", open=False):
                    synapse_monitor = gr.Number(label="Active Synapses")

        # Gradio 6.0: History is List[Dict[str, str]] with 'role' and 'content' keys
        def bot_response(message: str, history: List[Dict[str, str]]) -> Any:
            """
            ユーザー入力に対する応答処理と、脳状態の観測更新を行うコールバック。
            """
            # 履歴の初期化
            if history is None:
                history = []

            response_text = "..."
            observation: Dict[str, Any] = {}
            status_txt = "RUNNING"

            try:
                # 2. 会話エンジンの実行（思考プロセス）
                if message:
                    raw_res = chat_service.chat(message)
                    response_text = str(raw_res)

                # 3. 脳シミュレーションの実行 (1サイクル)
                dummy_input = torch.randn(1, 784)
                observation = brain.run_cycle(dummy_input)

            except Exception as e:
                logger.error(f"Execution Error: {e}")
                response_text = f"⚠️ SYSTEM ERROR: {e}"
                status_txt = "ERROR"
                observation = {}

            # 4. Gradio 6.0 形式で履歴を追加: メッセージ辞書形式
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response_text})

            # 5. データの安全な変換
            safe_observation = deep_safe_convert(observation)

            cycle_val = safe_observation.get("cycle", 0)
            status_txt = str(safe_observation.get("status", status_txt))
            phase_txt = str(safe_observation.get("phase", "Wake"))

            spikes_data = safe_observation.get("substrate_activity", {})
            bio_data = safe_observation.get("bio_metrics", {})
            synapse_val = safe_observation.get("synapse_count", 0)

            return (
                history,
                cycle_val,
                status_txt,
                phase_txt,
                spikes_data,
                bio_data,
                synapse_val,
            )

        # イベントハンドラの設定
        ui_outputs = [
            chatbot,
            cycle_monitor,
            status_monitor,
            phase_monitor,
            spikes_monitor,
            bio_monitor,
            synapse_monitor,
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

        # 入力完了時に入力欄をクリア
        msg.submit(lambda: "", None, msg)
        submit_btn.click(lambda: "", None, msg)

        # リセット処理
        def reset_system() -> Any:
            logger.info("System Reset Requested.")
            try:
                brain.boot()
            except Exception as e:
                logger.error(f"Reset failed: {e}")
            # 初期状態を返す (historyは空リスト)
            return [], 0, "RESET", "Wake", {}, {}, 0

        clear_btn.click(
            reset_system,
            inputs=None,
            outputs=ui_outputs,
        )

    return demo


def main() -> None:
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

    demo.queue().launch(
        server_name="127.0.0.1",
        share=False,
        debug=True,
        theme=gr.themes.Soft(),  # Gradio 6.0: themeはlaunch()に移動
    )


if __name__ == "__main__":
    main()
