# ファイルパス: app/deployment.py
# 日本語タイトル: SNN Inference Engine (Full Loop: Sense -> Think -> Act)
# 修正内容:
#   - SimpleMotorActuatorを導入し、Motor野の活動に基づく応答生成を実装。
#   - 仮の応答ロジックを廃止。

import torch
import logging
import time
from typing import Iterator, Tuple, Dict, Any, List, Optional
from omegaconf import DictConfig

# OS Core
from snn_research.core.neuromorphic_os import NeuromorphicOS
# I/O Modules
from snn_research.io.spike_encoder import TextSpikeEncoder
from snn_research.io.actuator import SimpleMotorActuator

logger = logging.getLogger(__name__)

class SNNInferenceEngine:
    """
    Neuromorphic OSのラッパー。
    [Input Text] -> Encoder -> [SNN Kernel] -> Motor Spikes -> Actuator -> [Output Text]
    """
    def __init__(self, brain: NeuromorphicOS, config: DictConfig):
        self.brain = brain
        self.config = config
        self.last_inference_stats: Dict[str, Any] = {}
        
        # 1. Sensory Encoder (Text -> Spikes)
        input_dim = self.brain.config.get("input_dim", 784)
        self.encoder = TextSpikeEncoder(
            num_neurons=input_dim, 
            device=str(self.brain.device)
        )
        
        # 2. Motor Actuator (Spikes -> Text)
        output_dim = self.brain.config.get("output_dim", 10)
        self.actuator = SimpleMotorActuator(output_dim=output_dim)
        
        logger.info("🤖 Inference Engine ready with Sensory-Motor loop.")

    def generate(
        self, 
        prompt: str, 
        max_len: int = 100, 
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None
    ) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """
        思考・行動生成ループ。
        """
        if stop_sequences is None:
            stop_sequences = []

        total_spikes = 0
        start_time = time.time()
        
        # 応答の蓄積用
        accumulated_response = ""
        last_action = ""

        # SNNは「状態」を持つため、入力が続いている間、少しずつ反応が変わる可能性がある
        # ここでは max_len 回のステップを実行し、Motor野が強く反応した時に言葉を発する
        
        step_interval = 10 # 何ステップごとにActuatorを確認するか

        for i in range(max_len):
            # --- 1. Sense: テキストからスパイク生成 ---
            # 持続的な入力として与える
            input_spikes_seq = self.encoder.forward(prompt, duration=1)
            input_tensor = input_spikes_seq.squeeze(1)

            # --- 2. Process: OSカーネル実行 ---
            cycle_result = self.brain.run_cycle(input_tensor)
            
            # --- 3. Observe: 内部状態の集計 ---
            substrate_state = cycle_result.get("substrate_state", {})
            current_spikes_dict = substrate_state.get("spikes", {})
            
            # 全スパイク数カウント
            step_spikes = 0
            for area_name, spikes in current_spikes_dict.items():
                if spikes is not None:
                    step_spikes += int(spikes.sum().item())
            total_spikes += step_spikes

            # --- 4. Act: 行動生成 (Motor野の読み取り) ---
            # 毎ステップ出力するとうるさいので、一定間隔または発火閾値で出力
            chunk = ""
            
            if i % step_interval == 0:
                # Motor野のスパイクを取得
                motor_spikes = current_spikes_dict.get("Motor")
                
                if motor_spikes is not None:
                    # Actuatorでデコード
                    action = self.actuator.decode(motor_spikes)
                    
                    # 無言(...) 以外で、かつ直前と同じ言葉でなければ出力
                    if action != "..." and action != last_action:
                        chunk = action + " "
                        last_action = action
                        accumulated_response += chunk

            # 初回のみフィードバック表示 (デモ用)
            if i == 0:
                chunk = "(Thinking...) "
            
            # ストリーミング用にyield
            stats = {
                "total_spikes": total_spikes,
                "step": i + 1,
                "step_spikes": step_spikes,
                "last_motor": last_action
            }
            
            yield chunk, stats
            
            # 停止条件
            if any(stop in accumulated_response for stop in stop_sequences):
                break
            
            # 少しWaitを入れてアニメーションさせる
            time.sleep(0.01)

        self.last_inference_stats = {
            "total_spikes": total_spikes,
            "duration": time.time() - start_time
        }