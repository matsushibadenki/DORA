# benchmarks/stability_benchmark.py
# Title: Stability Benchmark (Mypy Ignore Fix)
# Description: sklearnの型ヒント欠落を無視。

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
        
        for noise in noise_levels:
            noisy_X = X_tensor + torch.randn_like(X_tensor) * noise
            
            if hasattr(self.brain, "process_step"):
                outputs = []
                for i in range(min(100, len(noisy_X))):
                    sample = noisy_X[i].unsqueeze(0)
                    out = self.brain.process_step(sample)
                    
                    if isinstance(out, dict) and "output" in out:
                        feat = out["output"]
                    else:
                        feat = torch.zeros(1, 10)
                        
                    outputs.append(feat.detach().cpu().numpy().flatten())
                
                features = np.array(outputs)
                
                clf = LogisticRegression(max_iter=200)
                clf.fit(features, y[:100])
                acc = clf.score(features, y[:100])
                
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