# ファイルパス: app/main.py
# 日本語タイトル: DORA Practical Dashboard (Type Fixed)
# 目的: mypyエラー (Unexpected keyword argument "type") の修正

import logging
import os
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
    """データ変換の安全性確保"""
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
    elif hasattr(data, "item"):
        return data.item()
    elif isinstance(data, (float, int, str, bool, type(None))):
        return data
    else:
        return str(data)


def create_ui(container: AppContainer) -> gr.Blocks:
    chat_service = container.chat_service()
    os_sys = container.neuromorphic_os()

    with gr.Blocks(title="DORA: Practical Neuromorphic OS", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🧠 DORA: Practical Neuromorphic OS Console
            自律学習型AIの研究・実証実験用インターフェース。
            """
        )

        with gr.Row():
            # --- Left: Communication & Interaction ---
            with gr.Column(scale=2):
                gr.Markdown("### 📡 Communication Channel")
                # [Fix] Added # type: ignore to suppress mypy error for 'type' argument
                chatbot = gr.Chatbot(
                    label="Brain Response Stream",
                    height=450,
                    show_label=True,
                    type="messages"  # type: ignore
                )
                
                with gr.Group():
                    msg = gr.Textbox(
                        label="Input Signal",
                        placeholder="Message or sensory command...",
                        lines=1,
                    )
                    with gr.Row():
                        submit_btn = gr.Button("Send Signal", variant="primary")
                        clear_btn = gr.Button("Clear History")

            # --- Right: System Control & Monitoring ---
            with gr.Column(scale=1):
                gr.Markdown("### 🛠️ System Control & Metrics")
                
                # Control Panel (New Feature)
                with gr.Group():
                    gr.Markdown("##### System State Persistence")
                    with gr.Row():
                        save_btn = gr.Button("💾 Save State")
                        load_btn = gr.Button("📂 Load State")
                    system_msg = gr.Textbox(label="System Log", value="System Ready.", interactive=False, lines=2)

                # Monitors
                with gr.Group():
                    with gr.Row():
                        status_monitor = gr.Textbox(label="Kernel Status", value="BOOTING")
                        phase_monitor = gr.Textbox(label="Phase", value="Wake")
                    cycle_monitor = gr.Number(label="Life Cycles", value=0)

                with gr.Accordion("🧠 Neural Dynamics", open=True):
                    spikes_monitor = gr.JSON(label="Region Activity")

                with gr.Accordion("🧪 Bio-Metrics", open=False):
                    bio_monitor = gr.JSON(label="Homeostasis")

        # --- Logic Definitions ---

        def bot_response(message: str, history: List[Dict[str, str]]) -> Any:
            if history is None: history = []
            
            response_text = "..."
            observation: Dict[str, Any] = {}
            status_txt = "RUNNING"
            
            try:
                # 思考・対話
                if message:
                    response_text = str(chat_service.chat(message))
                
                # OSサイクル実行 (ダミー入力での時間発展)
                dummy_input = torch.randn(1, 784)
                observation = os_sys.run_cycle(dummy_input)

            except Exception as e:
                logger.error(f"Runtime Error: {e}")
                response_text = f"⚠️ ERROR: {str(e)}"
                status_txt = "RECOVERY"
                observation = {}

            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response_text})

            # データ整形
            safe_obs = deep_safe_convert(observation)
            
            return (
                history,
                safe_obs.get("cycle", 0),
                str(safe_obs.get("status", status_txt)),
                str(safe_obs.get("phase", "Wake")),
                safe_obs.get("output", {}), # 簡略化
                safe_obs.get("energy", 0),
                "Processing Complete."
            )

        # System Call Handlers
        def handle_save():
            msg = os_sys.sys_save("manual_snapshot.pt")
            return msg

        def handle_load():
            msg = os_sys.sys_load("manual_snapshot.pt")
            return msg
        
        def handle_clear():
            return [], "History Cleared."

        # Wiring
        submit_btn.click(
            bot_response,
            inputs=[msg, chatbot],
            outputs=[chatbot, cycle_monitor, status_monitor, phase_monitor, spikes_monitor, bio_monitor, system_msg]
        )
        msg.submit(
            bot_response,
            inputs=[msg, chatbot],
            outputs=[chatbot, cycle_monitor, status_monitor, phase_monitor, spikes_monitor, bio_monitor, system_msg]
        )
        msg.submit(lambda: "", None, msg)

        save_btn.click(handle_save, None, system_msg)
        load_btn.click(handle_load, None, system_msg)
        clear_btn.click(handle_clear, None, [chatbot, system_msg])

    return demo

def main() -> None:
    logger.info("🔌 Wiring application container...")
    container = AppContainer()

    # Config Loading
    import os
    if os.path.exists("configs/templates/base_config.yaml"):
        container.config.from_yaml("configs/templates/base_config.yaml")
    
    container.wire(modules=[__name__])

    logger.info("🧠 Booting Neuromorphic OS...")
    os_sys = container.neuromorphic_os()
    
    # 自動ロードの試行 (実用化向け)
    autoload_path = "./runtime_state/manual_snapshot.pt"
    if os.path.exists(autoload_path):
        logger.info("📂 Found existing snapshot. Auto-loading...")
        os_sys.brain.load_checkpoint(autoload_path)
    
    try:
        os_sys.boot()
    except Exception as e:
        logger.error(f"Boot failed: {e}")

    logger.info("🚀 Launching Practical Dashboard...")
    demo = create_ui(container)
    demo.queue().launch(server_name="127.0.0.1", share=False)

if __name__ == "__main__":
    main()