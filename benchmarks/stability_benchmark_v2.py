# benchmarks/stability_benchmark_v2.py
# Title: Stability Benchmark v2.1 (Energy Reset)
# Description: 
#   Run間に脳のエネルギーを回復させ、疲労によるテスト失敗を防ぐ。

import sys
import os
import argparse
import logging
import torch
import numpy as np
from pathlib import Path

# プロジェクトルートへのパス設定
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.containers import AppContainer
from benchmarks.stability_benchmark import StabilityBenchmark

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger("StabilityBenchV2")

def run_benchmark(runs: int = 1, epochs: int = 1):
    print("\n" + "="*60)
    print("🧪 DORA Stability Benchmark v2.1: Real Brain Testing")
    print("="*60 + "\n")

    # --- 1. Brain Initialization (Kernel Mode) ---
    logger.info("Initializing DORA Brain in Event-Driven Kernel Mode...")
    
    container = AppContainer()
    config_path = Path("configs/templates/base_config.yaml")
    if not config_path.exists():
        config_path = Path("configs/templates/base_config.yaml").resolve()
        if not config_path.exists():
            config_path = Path(__file__).resolve().parents[1] / "configs/templates/base_config.yaml"
    
    if config_path.exists():
        container.config.from_yaml(str(config_path))
    else:
        logger.warning("Base config not found. Using defaults.")

    # 強制的にKernelモード(CPU)にする
    container.config.training.paradigm.from_value("event_driven")
    container.config.device.from_value("cpu")
    
    os_kernel = container.neuromorphic_os()
    brain = os_kernel.brain
    os_kernel.boot()
    
    if brain.use_kernel and brain.kernel_substrate:
        logger.info(f"✅ Kernel Active. Neurons: {len(brain.kernel_substrate.kernel.neurons)}")
    else:
        logger.warning("⚠️ Kernel mode not active. Running in Matrix mode.")

    # --- 2. Run Benchmark ---
    
    overall_scores = []
    
    for r in range(runs):
        print(f"\n🔄 Run {r+1}/{runs}")
        
        # [Fix] Runの前にエネルギーを全回復させる
        if hasattr(brain, "astrocyte"):
            brain.astrocyte.replenish_energy(1000.0)
            brain.astrocyte.clear_fatigue(100.0)
            # logger.info("🔋 Brain energy replenished for next run.")
        
        # StabilityBenchmarkクラスを使用（実Brainを渡す）
        benchmark = StabilityBenchmark(brain)
        
        # ノイズレベルを変えてテスト (0%, 10%, 30%, 50%)
        results = benchmark.run_noise_robustness_test(noise_levels=[0.0, 0.1, 0.3, 0.5])
        
        # 平均スコアを計算
        avg_acc = np.mean(list(results.values()))
        overall_scores.append(avg_acc)
        
    print("\n" + "="*60)
    print("📊 Final Results Summary")
    print(f"   Runs: {runs}")
    print(f"   Average Stability Score: {np.mean(overall_scores):.4f}")
    print(f"   Standard Deviation: {np.std(overall_scores):.4f}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stability Benchmark on DORA Brain")
    parser.add_argument("--runs", type=int, default=1, help="Number of benchmark runs")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs")
    parser.add_argument("--threshold", type=int, default=90, help="Success threshold")
    
    args = parser.parse_args()
    
    run_benchmark(runs=args.runs, epochs=args.epochs)