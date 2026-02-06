# benchmarks/stability_benchmark.py
# Title: Stability Benchmark (Mypy Ignore)
# Description: sklearnの型ヒント欠落を無視。

import torch
import numpy as np
import logging
from typing import Dict, Any, List, Optional

# [Fix] Ignore missing type stubs for sklearn
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
from sklearn.datasets import load_digits # type: ignore

from snn_research.core.snn_core import SpikingNeuralSubstrate

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
            
            # Simple simulation: Brain process
            # Assuming brain has a method to get embedding/output
            if hasattr(self.brain, "process_step"):
                # Use subset for speed
                outputs = []
                for i in range(min(100, len(noisy_X))):
                    sample = noisy_X[i].unsqueeze(0)
                    out = self.brain.process_step(sample)
                    
                    # Extract feature vector from output dictionary
                    if isinstance(out, dict) and "output" in out:
                        feat = out["output"]
                    else:
                        feat = torch.zeros(1, 10) # Fallback
                        
                    outputs.append(feat.detach().cpu().numpy().flatten())
                
                features = np.array(outputs)
                
                # Evaluate separability (using LogReg as probe)
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
    # Dummy mock
    class MockBrain:
        def process_step(self, x):
            return {"output": x} # Pass-through
            
    bench = StabilityBenchmark(MockBrain())
    bench.run_noise_robustness_test()