# ファイルパス: scripts/experiments/brain/run_stability_validation.py
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import torch
import time
import logging
import numpy as np
from pathlib import Path
from typing import Tuple

from snn_research.models.visual_cortex import VisualCortex as VisualCortexV2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StabilityValidation")

def generate_dummy_data(batch_size: int = 64, dim: int = 784) -> Tuple[torch.Tensor, torch.Tensor]:
    # Positive: Strong structured pattern (First 200 pixels active)
    pos_data = torch.rand(batch_size, dim) * 0.1
    pos_data[:, :200] += 2.0 
    
    # Negative: Pure random noise
    neg_data = torch.rand(batch_size, dim) * 1.5
    
    return pos_data, neg_data

def run_experiment():
    logger.info("🧪 Starting Phase 2 Stability Validation (Correction V3)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tuned Config
    config = {
        "input_dim": 784,
        "hidden_dim": 1024,
        "num_layers": 3,
        "dt": 1.0,
        "tau_mem": 5.0,
        "learning_rate": 0.05,
        "ff_threshold": 3.0, # Lower threshold for trace-based goodness
        "noise_level": 1.5
    }
    
    brain = VisualCortexV2(device, config).to(device)
    logger.info(f"Brain V2 initialized on {device}")

    history = {
        "latency": [],
        "pos_goodness": [],
        "neg_goodness": [],
        "stability_score": []
    }

    # --- Phase 1: Learning ---
    logger.info("☀️  Wake Phase: Learning patterns...")
    iterations = 400 # Slightly more iterations
    
    start_time = time.time()
    for i in range(iterations):
        pos_x, neg_x = generate_dummy_data(batch_size=32)
        pos_x, neg_x = pos_x.to(device), neg_x.to(device)

        # Wake (Positive)
        brain.reset_state()
        brain(pos_x, phase="wake")
        
        # Sleep (Negative / Contrastive)
        brain.reset_state()
        brain(neg_x, phase="sleep")
        
        if (i + 1) % 100 == 0:
            metrics = brain.get_stability_metrics()
            rate = metrics.get("V1_firing_rate", 0.0)
            logger.info(f"  Step {i+1}: V1 Rate={rate:.2%} | V1 W_Std={metrics.get('V1_weight_std', 0):.4f}")

    # --- Phase 2: Validation ---
    logger.info("🔍 Validation Phase...")
    
    num_tests = 100
    for i in range(num_tests):
        pos_x, neg_x = generate_dummy_data(batch_size=1)
        pos_x, neg_x = pos_x.to(device), neg_x.to(device)

        t0 = time.perf_counter()
        
        # Test Positive
        brain.reset_state()
        brain(pos_x, phase="inference")
        pos_stats = brain.get_goodness()
        
        # Test Negative
        brain.reset_state()
        brain(neg_x, phase="inference")
        neg_stats = brain.get_goodness()
            
        dt_ms = (time.perf_counter() - t0) * 1000
        history["latency"].append(dt_ms)
        
        # Score using V2 (deeper features) or V3
        score_pos = pos_stats.get("V2_goodness", 0) + pos_stats.get("V3_goodness", 0)
        score_neg = neg_stats.get("V2_goodness", 0) + neg_stats.get("V3_goodness", 0)
        
        history["pos_goodness"].append(score_pos)
        history["neg_goodness"].append(score_neg)
        
        # Stability: Positive should have higher "Goodness" (Energy)
        if score_pos > score_neg:
            history["stability_score"].append(1.0)
        else:
            history["stability_score"].append(0.0)

    # --- Report ---
    avg_latency = np.mean(history["latency"])
    stability = np.mean(history["stability_score"]) * 100
    pos_g = np.mean(history["pos_goodness"])
    neg_g = np.mean(history["neg_goodness"])

    print("\n" + "="*50)
    print("📊 Phase 2 Validation Report (Final)")
    print("="*50)
    print(f"Inference Latency : {avg_latency:.4f} ms  (Target: < 10ms) {'✅' if avg_latency < 10 else '❌'}")
    print(f"Learning Stability: {stability:.2f} %     (Target: > 95%)  {'✅' if stability >= 95 else '⚠️'}")
    print(f"Signal Separation : Pos={pos_g:.2f} vs Neg={neg_g:.2f}")
    
    metrics = brain.get_stability_metrics()
    print("-" * 30)
    print("Internal Dynamics:")
    for k, v in metrics.items():
        if "firing_rate" in k:
             print(f"  {k}: {v:.2%}")
        elif "weight_mean" in k:
             print(f"  {k}: {v:.4f}")
    print("="*50 + "\n")

    # Save
    report_path = Path("workspace/reports/validation_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(f"Latency: {avg_latency}\nStability: {stability}\n")

if __name__ == "__main__":
    run_experiment()