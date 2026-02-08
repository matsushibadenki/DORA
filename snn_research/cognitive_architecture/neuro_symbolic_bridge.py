# directory: snn_research/cognitive_architecture
# file: neuro_symbolic_bridge.py
# purpose: 直感(System 1)と論理(System 2)を調停するニューロシンボリック・ブリッジの実装

import torch
import time
from typing import Any, Dict, Optional, Tuple

# SARA Adapterのインポート（System 1として使用）
try:
    from snn_research.models.adapters.sara_adapter import SARAAdapter
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    from snn_research.models.adapters.sara_adapter import SARAAdapter

class System2Oracle:
    """
    System 2（論理推論・高度な知識）のシミュレーター
    現実のユースケースでは、ここがLLM (GPT-4) や 形式的推論エンジン に置き換わります。
    """
    def __init__(self, cost_per_call: float = 10.0):
        self.cost_per_call = cost_per_call  # System 2の起動コスト（エネルギー/時間）
        self.total_energy_used = 0.0

    def query(self, input_data: Any, correct_label: int) -> Dict[str, Any]:
        """
        高コストな推論を実行し、正確な答えと「理由」を返す
        """
        # シミュレーションされた思考時間（重い処理を表現）
        time.sleep(0.5)
        
        self.total_energy_used += self.cost_per_call
        
        return {
            "answer": correct_label,
            "reasoning": f"Deep analysis confirms features matching class {correct_label}.",
            "confidence": 1.0
        }

class NeuroSymbolicBridge:
    """
    System 1 (SARA) と System 2 (Oracle/LLM) の調停役
    不確実性（Uncertainty）に基づいて処理ルートを動的に切り替える（メタ認知）。
    """
    def __init__(self, system1_agent: SARAAdapter, confidence_threshold: float = 0.85):
        self.system1 = system1_agent
        self.system2 = System2Oracle()
        self.threshold = confidence_threshold
        
        # 統計情報
        self.stats = {
            "s1_calls": 0,
            "s2_calls": 0,
            "autonomous_learning_events": 0
        }

    def process_input(self, input_data: Any, correct_label: Optional[int] = None) -> Dict[str, Any]:
        """
        入力に対する統合的な推論・学習プロセス
        """
        # --- Step 1: System 1 (Fast & Cheap) ---
        start_time = time.time()
        s1_result = self.system1.think(input_data)
        s1_latency = (time.time() - start_time) * 1000
        
        self.stats["s1_calls"] += 1
        
        pred = s1_result["prediction"]
        conf = s1_result["confidence"]
        
        # --- Step 2: Meta-Cognitive Check ---
        # 確信度が閾値以上なら、System 1の答えを採用（省エネ）
        if conf >= self.threshold:
            return {
                "source": "System 1 (Intuition)",
                "prediction": pred,
                "confidence": conf,
                "latency": s1_latency,
                "energy_cost": 1.0, # SNNの低コスト
                "explanation": "Intuitive match found."
            }
            
        # --- Step 3: System 2 (Slow & Expensive) ---
        # 確信度が低い、かつ正解ラベル（環境からのフィードバックや教師）にアクセス可能な場合
        # 現実では「人間に聞く」や「LLMに投げる」アクションに相当
        if correct_label is not None:
            print(f"  [Meta-Cognition] Uncertainty detected (Conf: {conf:.1%}). Waking up System 2...")
            
            s2_result = self.system2.query(input_data, correct_label)
            self.stats["s2_calls"] += 1
            
            s2_answer = s2_result["answer"]
            
            # --- Step 4: Active Learning (Plasticity) ---
            # System 2の答えを使ってSystem 1を即時教育する
            print(f"  [Plasticity] Teaching System 1: {s2_answer} (Reason: {s2_result['reasoning']})")
            loss = self.system1.learn_instance(input_data, s2_answer, max_steps=10)
            self.stats["autonomous_learning_events"] += 1
            
            return {
                "source": "System 2 (Reasoning)",
                "prediction": s2_answer,
                "confidence": 1.0,
                "latency": s1_latency + 500.0, # S2の遅延を加算
                "energy_cost": 100.0, # 高コスト
                "explanation": s2_result["reasoning"],
                "learning_loss": loss
            }
        
        # System 2が使えない場合はSystem 1の不確実な答えを返す
        return {
            "source": "System 1 (Uncertain)",
            "prediction": pred,
            "confidence": conf,
            "latency": s1_latency,
            "energy_cost": 1.0,
            "explanation": "Low confidence, but no teacher available."
        }

    def print_stats(self):
        print("\n=== Neuro-Symbolic Bridge Statistics ===")
        print(f"System 1 Calls (Fast): {self.stats['s1_calls']}")
        print(f"System 2 Calls (Slow): {self.stats['s2_calls']}")
        print(f"Learning Events      : {self.stats['autonomous_learning_events']}")
        print(f"System 2 Energy Used : {self.system2.total_energy_used}")
        print("========================================")