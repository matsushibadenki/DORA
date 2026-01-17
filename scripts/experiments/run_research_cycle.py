# ファイルパス: scripts/experiments/run_research_cycle.py
# 日本語タイトル: Research Cycle Experiment Runner (With History & Tuning)
# 目的・内容:
#   Neuromorphic OSの標準実験スクリプト。
#   [修正] 全サイクルの時系列データを記録し、実験終了後にJSONへ保存する機能を追加。
#   [修正] 意識が発生しやすいよう、入力感度とパラメータを調整。

import logging
import time
import os
import sys
import json
import torch
from torchvision import datasets, transforms # type: ignore

# ---------------------------------------------------------
# [Setup] パス設定
# ---------------------------------------------------------
print("🚀 Initializing Experiment Environment...")
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

sys.dont_write_bytecode = True

# ---------------------------------------------------------
# [Log Config] 強制ログ設定
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(levelname)s | %(message)s',
    force=True
)
logger = logging.getLogger("Experiment")

# ---------------------------------------------------------
# [Import] コアモジュール
# ---------------------------------------------------------
try:
    print("⏳ Importing NeuromorphicOS Kernel...")
    from snn_research.core.neuromorphic_os import NeuromorphicOS
    print("✅ Kernel imported successfully.")
except ImportError as e:
    print(f"❌ Critical Import Error: {e}")
    sys.exit(1)

def load_mnist_sample(batch_size=32):
    """実験用の感覚入力としてMNISTを使用"""
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def run_experiment():
    print("\n🧪 >>> STARTING MAIN EXPERIMENT LOOP (With Data Recording) <<<\n")
    logger.info("Starting Neuromorphic Research Cycle Experiment...")
    
    # 1. OSの構成設定（意識が出やすいよう調整）
    config = {
        "input_dim": 784,
        "hidden_dim": 512, 
        "hippocampus_dim": 256,
        "output_dim": 10,
        "max_energy": 2000.0,
        # 閾値を少し調整できる設計であればここで指定（現状はコード内固定）
    }
    
    # 2. カーネルの起動
    try:
        os_kernel = NeuromorphicOS(config, device_name="auto")
        os_kernel.boot()
        print(f"🖥️ Kernel booted on: {os_kernel.device}")
    except Exception as e:
        logger.error(f"❌ Boot Failed: {e}")
        raise e
    
    # 3. データソースの準備
    print("📦 Loading sensory data (MNIST)...")
    try:
        data_loader = load_mnist_sample(batch_size=16)
        data_iterator = iter(data_loader)
    except Exception as e:
        logger.warning(f"⚠️ Failed to load MNIST: {e}. Switching to noise input.")
        data_iterator = None

    # 4. 実験ループ設定
    total_cycles = 600 # サイクル数を少し増やす
    wake_duration = 150
    sleep_duration = 50
    
    cycle_counter = 0
    phase = "wake"
    phase_timer = 0
    
    # ★時系列データ保存用リスト
    history = []
    
    print(f"⏱️ Experiment Start: {total_cycles} cycles planned.")
    
    try:
        while cycle_counter < total_cycles:
            cycle_counter += 1
            phase_timer += 1
            
            # --- Phase Control ---
            energy_level = 1.0
            if hasattr(os_kernel.astrocyte, "current_energy"):
                e_curr = os_kernel.astrocyte.current_energy
                e_max = os_kernel.astrocyte.max_energy
                energy_level = e_curr / e_max

            if phase == "wake":
                if phase_timer >= wake_duration or energy_level < 0.15: # 限界まで粘る
                    msg = f"🌙 [Cycle {cycle_counter}] Falling Asleep... (Energy: {energy_level*100:.1f}%)"
                    print(msg)
                    logger.info(msg)
                    phase = "sleep"
                    phase_timer = 0
            
            elif phase == "sleep":
                if phase_timer >= sleep_duration and energy_level > 0.95: # 十分回復するまで寝る
                    msg = f"☀️ [Cycle {cycle_counter}] Waking Up! (Energy: {energy_level*100:.1f}%)"
                    print(msg)
                    logger.info(msg)
                    phase = "wake"
                    phase_timer = 0

            # --- Input Generation ---
            if phase == "wake" and data_iterator:
                try:
                    images, _ = next(data_iterator)
                except StopIteration:
                    data_loader = load_mnist_sample(batch_size=16)
                    data_iterator = iter(data_loader)
                    images, _ = next(data_iterator)
                
                # 入力を少し強調（コントラストを上げる）して意識を刺激する
                sensory_input = images.view(images.size(0), -1) * 2.0 
            else:
                sensory_input = torch.zeros(16, 784)

            # --- Run OS Cycle ---
            observation = os_kernel.run_cycle(sensory_input, phase=phase)
            
            # ★履歴に追加 (Tensorなどはfloatに変換済みであることを期待)
            history.append(observation)
            
            # --- Live Monitoring ---
            if cycle_counter % 10 == 0:
                bio = observation["bio_metrics"]
                spikes = observation["substrate_activity"]
                avg_act = sum(spikes.values()) / len(spikes) if spikes else 0.0
                
                print(
                    f"Cycle {cycle_counter:03d} | {phase.upper()} | "
                    f"Energy: {bio.get('current_energy', 0):.0f} | "
                    f"Act: {avg_act:.4f} | "
                    f"Conscious: {observation['consciousness_level']:.4f}"
                )
                
            time.sleep(0.005) # 高速化

    except KeyboardInterrupt:
        print("\n⚠️ Experiment interrupted manually.")
    except Exception as e:
        logger.error(f"❌ Runtime Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        os_kernel.shutdown()
        
        # ★時系列データの保存
        history_path = os.path.join("runtime_state", "experiment_history.json")
        try:
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)
            print(f"💾 Full experiment history saved to: {history_path}")
            print(f"📊 Run 'python scripts/visualization/plot_research_data.py' to visualize.")
        except Exception as e:
            print(f"❌ Failed to save history: {e}")

if __name__ == "__main__":
    run_experiment()