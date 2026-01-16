# ファイルパス: snn_research/io/actuator.py
# 日本語タイトル: Motor Actuator (Spike to Text Decoder)
# 目的・内容:
#   - 運動野(Motor Cortex)のスパイク活動を、意味のあるテキストアクションに変換する。
#   - 単純なWinner-Take-All方式で、最も発火したニューロンに対応する概念を出力する。

import torch
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class SimpleMotorActuator:
    """
    運動野のスパイクを読み取り、事前に定義されたアクション（言葉）に変換するクラス。
    """
    def __init__(self, output_dim: int = 10):
        self.output_dim = output_dim
        
        # 10個のニューロンに対応する「概念」または「反応」の定義
        # 将来的には学習によって獲得されるべきだが、初期段階としてハードコードする
        self.concept_map = {
            0: "I see.",           # 受容
            1: "Interesting.",     # 興味
            2: "I am not sure.",   # 疑問
            3: "Yes!",             # 肯定/興奮
            4: "No.",              # 否定
            5: "Tell me more.",    # 探求
            6: "Processing...",    # 思考中
            7: "I feel something.",# 感覚
            8: "Analyzing.",       # 分析
            9: "Wait."             # 抑制
        }
        
        logger.info(f"🦾 Motor Actuator initialized. Mapping {output_dim} neurons to concepts.")

    def decode(self, motor_spikes: torch.Tensor) -> str:
        """
        スパイク列（またはレート）を受け取り、アクション文字列を返す。
        
        Args:
            motor_spikes (Tensor): (Batch, Time, Neurons) or (Batch, Neurons)
        """
        # バッチサイズ1を想定
        if motor_spikes.dim() == 3:
            # 時間方向に合計して発火数をカウント (Batch, Neurons)
            activity = motor_spikes.sum(dim=1)
        else:
            activity = motor_spikes

        # activity: (Batch, Neurons) -> (Neurons)
        activity = activity.squeeze(0)
        
        # 全く発火していない場合
        if activity.sum() == 0:
            return "..."

        # 最も発火したニューロンのインデックスを取得 (Winner-Take-All)
        winner_idx = torch.argmax(activity).item()
        
        # 発火の強さ（確信度）
        confidence = activity[winner_idx].item()
        
        # マッピングから応答を取得
        response = self.concept_map.get(winner_idx, "?")
        
        return response

    def get_status(self, motor_spikes: torch.Tensor) -> Dict[str, float]:
        """デバッグ用: 各概念の活性度を返す"""
        if motor_spikes.dim() == 3:
            activity = motor_spikes.sum(dim=1).squeeze(0)
        else:
            activity = motor_spikes.squeeze(0)
            
        status = {}
        for idx, text in self.concept_map.items():
            if idx < len(activity):
                status[text] = activity[idx].item()
        return status