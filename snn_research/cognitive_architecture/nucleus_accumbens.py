# snn_research/cognitive_architecture/nucleus_accumbens.py
# Title: Nucleus Accumbens (Reward System)
# Description: 
#   ユーザーの言葉から感情的フィードバック(報酬/罰)を抽出する。
#   - Positive: "Good", "Great", "Thanks", "Yes" -> Dopamine Release (+1.0)
#   - Negative: "Bad", "No", "Wrong", "Stop"     -> Dopamine Dip (-1.0)
#   この信号を用いて、海馬の記憶の重み(Confidence)を更新する。

import logging

class NucleusAccumbens:
    def __init__(self, brain):
        self.brain = brain
        self.logger = logging.getLogger("NucleusAccumbens")
        
        # Reward Keywords
        self.positive_rewards = ["GOOD", "GREAT", "EXCELLENT", "THANKS", "YES", "WELL DONE", "SMART"]
        self.negative_rewards = ["BAD", "NO", "WRONG", "STOP", "MISTAKE", "FALSE", "STUPID"]
        
        self.logger.info("🍬 Nucleus Accumbens initialized. Ready for dopamine.")

    def process_reward(self, text):
        """
        テキストを分析し、報酬値を返す。
        Returns:
            float: +1.0 (Reward), -1.0 (Punishment), 0.0 (Neutral)
        """
        text_upper = text.upper()
        
        # Check Positive
        if any(w in text_upper for w in self.positive_rewards):
            print(f"   🍬 [NucleusAccumbens] DOPAMINE SURGE! Reward detected.")
            return 1.0
            
        # Check Negative
        if any(w in text_upper for w in self.negative_rewards):
            print(f"   💀 [NucleusAccumbens] DOPAMINE DIP... Punishment detected.")
            return -1.0
            
        return 0.0