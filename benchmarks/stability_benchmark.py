# benchmarks/stability_benchmark.py
# Title: Stability Benchmark v1.1 (Fatigue Safe)
# Description: 脳が疲労(None)を返した場合のクラッシュを防ぐ修正を追加。

import torch
import numpy as np
import logging
from typing import Dict, Any, List, Optional

# [Fix] Correct type ignore syntax
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
from sklearn.datasets import load_digits # type: ignore

logger = logging.getLogger("StabilityBenchmark")

class StabilityBenchmark:
    def __init__(self, brain_system: Any):
        self.brain = brain_system
        self.metrics: Dict[str, float] = {}
        
    def run_noise_robustness_test(self, noise_levels: List[float] = [0.0, 0.1, 0.3, 0.5]) -> Dict[str, float]:
        print("⚡ Running Noise Robustness Benchmark...")
        results = {}
        
        # Load simple dataset
        data = load_digits()
        X, y = data.data, data.target
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # 出力次元の推定（初期化用）
        output_dim = 10
        if hasattr(self.brain, "d_model"):
            output_dim = self.brain.d_model
        
        for noise in noise_levels:
            noisy_X = X_tensor + torch.randn_like(X_tensor) * noise
            
            if hasattr(self.brain, "process_step"):
                outputs = []
                # テストサンプル数を制限 (100サンプル)
                limit = min(100, len(noisy_X))
                
                for i in range(limit):
                    sample = noisy_X[i].unsqueeze(0)
                    out = self.brain.process_step(sample)
                    
                    feat = None
                    if isinstance(out, dict) and "output" in out:
                        feat = out["output"]
                    
                    # [Fix] Handle None output (Fatigue/Sleep)
                    if feat is None:
                        # 疲労時はゼロベクトル（反応なし）として扱う
                        feat = torch.zeros(1, output_dim)
                    
                    # Ensure tensor is on CPU
                    if isinstance(feat, torch.Tensor):
                        feat = feat.detach().cpu()
                    
                    outputs.append(feat.numpy().flatten())
                
                features = np.array(outputs)
                
                # Check for NaNs or all zeros (Brain dead)
                if np.all(features == 0):
                    logger.warning(f"⚠️ Brain returned all zeros for noise {noise}. Probably fatigued.")
                    acc = 0.0
                else:
                    clf = LogisticRegression(max_iter=200)
                    # Use only valid features for training simple readout
                    clf.fit(features, y[:limit])
                    acc = clf.score(features, y[:limit])
                
                results[f"noise_{noise}"] = acc
                print(f"   Noise {noise}: Accuracy {acc:.3f}")
            else:
                logger.warning("Brain system does not support process_step.")
                
        self.metrics.update(results)
        return results

if __name__ == "__main__":
    class MockBrain:
        def process_step(self, x):
            return {"output": x} 
            
    bench = StabilityBenchmark(MockBrain())
    bench.run_noise_robustness_test()