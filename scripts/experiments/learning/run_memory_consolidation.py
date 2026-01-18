# ファイルパス: scripts/experiments/learning/run_memory_consolidation.py
# 日本語タイトル: Memory Consolidation Experiment (Structural Plasticity)
# 目的・内容:
#   Wake/Sleepサイクルを実行し、記憶学習を行う。
#   修正: シナプス数(Syn)の変動を監視し、脳の構造変化を記録する。

import logging
import time
import os
import sys
import json
import torch
import numpy as np
from collections import deque
from torchvision import datasets, transforms  # type: ignore

print("🚀 Initializing Memory Experiment Environment...")
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

sys.dont_write_bytecode = True

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s', force=True)
logger = logging.getLogger("MemoryExp")

try:
    from snn_research.core.neuromorphic_os import NeuromorphicOS
    print("✅ NeuromorphicOS imported.")
except ImportError as e:
    print(f"❌ Critical Import Error: {e}")
    sys.exit(1)


class PrototypeReadout:
    def __init__(self, num_classes=10, feature_dim=256):
        self.num_classes = num_classes
        self.prototypes = torch.zeros(num_classes, feature_dim)
        self.counts = torch.zeros(num_classes)

    def update(self, features, label):
        features = features.cpu().view(-1)
        if features.shape[0] != self.prototypes.shape[1]:
            return
        alpha = 0.1
        if self.counts[label] == 0:
            self.prototypes[label] = features
        else:
            self.prototypes[label] = (1 - alpha) * \
                self.prototypes[label] + alpha * features
        self.counts[label] += 1

    def predict(self, features):
        if self.counts.sum() == 0:
            return -1
        features = features.cpu().view(1, -1)
        f_norm = features / (features.norm() + 1e-8)
        p_norm = self.prototypes / \
            (self.prototypes.norm(dim=1, keepdim=True) + 1e-8)
        similarities = torch.matmul(p_norm, f_norm.t())
        return torch.argmax(similarities).item()


def load_mnist_sample(batch_size=32):
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST(data_dir, train=True,
                             download=True, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)
    return loader


def run_memory_experiment():
    print("\n🧠 >>> STARTING MEMORY CONSOLIDATION EXPERIMENT (With Structural Plasticity) <<<\n")

    config = {
        "input_dim": 784,
        "hidden_dim": 256,
        "hippocampus_dim": 512,
        "output_dim": 10,
        "max_energy": 2500.0,
    }

    os_kernel = NeuromorphicOS(config, device_name="auto")
    os_kernel.boot()
    print(f"🖥️ Kernel booted on: {os_kernel.device}")

    readout = PrototypeReadout(
        num_classes=10, feature_dim=config["hidden_dim"])
    data_loader = load_mnist_sample(batch_size=1)
    data_iterator = iter(data_loader)

    schedule = [
        {"phase": "wake",  "end_cycle": 300, "label": "Learning Phase 1"},
        {"phase": "sleep", "end_cycle": 450,
            "label": "Memory Consolidation (Pruning/Growth)"},
        {"phase": "wake",  "end_cycle": 800,
            "label": "Learning Phase 2 (Recall)"}
    ]

    total_cycles = 800
    cycle_counter = 0
    history = []
    recent_accuracy = deque(maxlen=20)

    try:
        while cycle_counter < total_cycles:
            cycle_counter += 1

            # --- Schedule ---
            current_phase = "wake"
            phase_label = ""
            for s in schedule:
                if cycle_counter <= s["end_cycle"]:
                    current_phase = s["phase"]
                    phase_label = s["label"]
                    break

            # --- Input ---
            target_label = -1
            sensory_input = torch.zeros(1, 784)

            if current_phase == "wake":
                try:
                    images, labels = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(data_loader)
                    images, labels = next(data_iterator)
                sensory_input = images.view(1, -1) * 2.5
                target_label = labels.item()

            # --- Run OS ---
            observation = os_kernel.run_cycle(
                sensory_input, phase=current_phase)

            # --- Readout & Reward ---
            raw_spikes = None
            if hasattr(os_kernel, "last_substrate_state"):
                spikes_dict = os_kernel.last_substrate_state.get("spikes", {})
                raw_spikes = spikes_dict.get("Association")

            if raw_spikes is None:
                raw_spikes = torch.zeros(
                    1, config["hidden_dim"]).to(os_kernel.device)
            else:
                raw_spikes = raw_spikes.float().detach()

            prediction = -1
            is_correct = 0

            if current_phase == "wake":
                prediction = readout.predict(raw_spikes)
                if prediction == target_label:
                    is_correct = 1
                    os_kernel.reward(amount=1.5)
                readout.update(raw_spikes, target_label)
                recent_accuracy.append(is_correct)

            # --- Recording ---
            acc_val = sum(recent_accuracy) / \
                len(recent_accuracy) if recent_accuracy else 0.0

            record = {
                "cycle": cycle_counter,
                "phase": current_phase,
                "phase_label": phase_label,
                "energy": observation["bio_metrics"]["energy"],
                "dopamine": observation["bio_metrics"].get("dopamine", 0.0),
                "synapse_count": observation.get("synapse_count", 0),  # ★追加
                "consciousness": observation["consciousness_level"],
                "accuracy": acc_val,
                "target": target_label,
                "prediction": prediction
            }
            history.append(record)

            if cycle_counter % 20 == 0:
                # ログにシナプス数(Syn)を追加
                syn = record['synapse_count']
                print(
                    f"Cycle {cycle_counter:03d} | {phase_label[:15]}.. | "
                    f"E: {record['energy']:.0f} | DA: {record['dopamine']:.2f} | "
                    f"Syn: {syn} | Acc: {acc_val:.2f}"
                )

            sleep_time = 0.001 if current_phase == "sleep" else 0.005
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n⚠️ Experiment interrupted.")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        os_kernel.shutdown()

        out_path = os.path.join(
            "runtime_state", "memory_experiment_history.json")
        try:
            with open(out_path, "w") as f:
                json.dump(history, f, indent=2)
            print(f"💾 Learning history saved to: {out_path}")
            print(
                "📊 Run 'python scripts/visualization/plot_memory_learning.py' to visualize.")
        except Exception:
            print("❌ Failed to save history.")


if __name__ == "__main__":
    run_memory_experiment()
