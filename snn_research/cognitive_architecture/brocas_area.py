# snn_research/cognitive_architecture/brocas_area.py
# Title: Broca's Area (Neural Gated Speech)
# Description: 
#   脳の興奮レベル(Spike Count)を監視し、閾値を超えた場合のみ発話を行う。
#   発話内容は、興奮度(Arousal)に基づいてトーン調整される。
#   - Silent (0 spikes): 無視 (...)
#   - Low (< 10): 冷静な応答 (Calm)
#   - High (> 15): 興奮した応答 (Excited)

import logging
import random

class BrocasArea:
    def __init__(self, brain):
        self.brain = brain
        self.logger = logging.getLogger("BrocasArea")
        self.threshold = 12.0 # MotorCortexと同じ閾値を使用
        self.logger.info("🗣️ Broca's Area initialized. Ready to speak.")

    def generate_response(self, input_text, spike_activity):
        """
        脳の反応に基づいて応答を生成する。
        脳が反応しなければ、DORAは言葉を発しない。
        """
        avg_spikes = sum(spike_activity) / len(spike_activity) if spike_activity else 0
        
        # 1. Neural Gating (脳が反応していないなら無視)
        if avg_spikes < 1.0:
            return None # 完全無視
        
        # 2. Tone Analysis (興奮レベルによるトーン変化)
        if avg_spikes > self.threshold:
            tone = "EXCITED"
            prefix = "⚡ [SHOUT] "
        else:
            tone = "CALM" # 今回の閾値設定ではここには来ない(0か19かのため)
            prefix = "💬 [SAY] "

        # 3. Simple Response Generation (本来はここでLLMを使うが、今回はルールベースで模倣)
        response = self._synthesize_text(input_text, tone)
        
        self.logger.info(f"   🧠 Brain Activity: {avg_spikes:.2f} -> Tone: {tone}")
        return f"{prefix}{response}"

    def _synthesize_text(self, input_text, tone):
        # 簡易応答ロジック
        if tone == "EXCITED":
            if "FIRE" in input_text.upper():
                return "DETECTED EMERGENCY! EVACUATING IMMEDIATELY!"
            elif "DANGER" in input_text.upper():
                return "DANGER SIGNAL RECEIVED! SYSTEMS ON HIGH ALERT!"
            else:
                return "ATTENTION! I AM RESPONDING TO STRONG STIMULI!"
        else:
            return f"I acknowledge: '{input_text}'."