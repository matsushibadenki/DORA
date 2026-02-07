# snn_research/cognitive_architecture/motor_cortex.py
# Title: Motor Cortex (Circular Import Fixed)
# Description: 
#   循環参照の原因となっていた 'from app.containers import AppContainer' を削除。
#   機能は前回のPrint Debug版を維持。

import logging

class MotorCortex:
    def __init__(self, brain=None, actuators=None, device='cpu', threshold=50.0):
        """
        brain: DORAの脳インスタンス
        actuators: アクチュエータのリスト
        device: 実行デバイス
        threshold: 反射行動の閾値
        """
        self.brain = brain
        self.actuators = actuators if actuators else []
        self.device = device
        self.threshold = threshold
        
        # ロガー設定 (標準出力で確認したい場合はprintを使用)
        self.logger = logging.getLogger("MotorCortex")
        print(f"🦾 [MotorCortex] Initialized. Threshold={self.threshold}")

    def monitor_and_act(self, spike_history):
        """
        直近の脳活動(スパイク履歴)を分析し、必要なら行動する
        """
        # 平均活動レベルを計算
        avg_activity = sum(spike_history) / len(spike_history) if spike_history else 0
        
        action = "IDLE"
        reaction = "💤 Idling..."

        # 判定ロジック
        if avg_activity > self.threshold:
            action = "ESCAPE"
            reaction = "🏃💨 EMERGENCY EVACUATION! (Running away)"
        elif avg_activity > (self.threshold * 0.5):
            action = "ALERT"
            reaction = "👀 LOOK AROUND (Alerted)"
        
        print(f"   🧠 [MotorCortex] Activity: {avg_activity:.2f} / Thr: {self.threshold} -> Action: {action}")
        return reaction

    def _trigger_reflex(self, action_type):
        # 内部メソッド（必要に応じて拡張可能）
        pass