# ファイルパス: scripts/experiments/run_research_cycle.py
# 日本語タイトル: Long-term Research Experiment Runner
# 目的・内容:
#   Neuromorphic OSを長時間（数千サイクル）動作させ、
#   Active Inferenceによる学習効果や、睡眠による構造変化（シナプス刈り込み）を
#   定量データとして記録する。

import sys
import os
import time
import json
import logging
import torch
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# プロジェクトルートへのパス設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.containers import AppContainer
from snn_research.io.spike_encoder import TextSpikeEncoder

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Experiment")


def run_experiment(cycles: int = 1000):
    """
    自律的な学習・睡眠サイクル実験を実行する。
    """
    # 1. セットアップ
    # 1. セットアップ
    container = AppContainer()
    # brain = container.brain()
    # brain.boot()
    os_sys = container.neuromorphic_os()
    os_sys.boot()

    # 入力エンコーダ（実験用刺激生成）
    encoder = TextSpikeEncoder(num_neurons=784, device=str(os_sys.device))

    # 学習させる概念（パターンの繰り返し提示）
    concepts = ["Apple", "Danger", "Food", "Shelter"]
    current_concept_idx = 0

    history = []

    logger.info(f"🧪 Starting experiment for {cycles} cycles...")

    # 2. メインループ
    pbar = tqdm(range(cycles))
    for i in pbar:
        # --- Context / Environment ---
        # スケジューラから現在のフェーズ（Wake/Sleep）を取得
        # (OS内部で自動遷移するが、実験のために強制力を働かせることも可能)
        # ここではOSの自律判断に任せる
        # phase = brain.scheduler.get_current_phase()
        phase = "wake" if os_sys.brain.is_awake else "sleep"

        input_tensor = torch.zeros(1, 784).to(os_sys.device)

        if phase == "wake":
            # 概念の切り替え (一定間隔で環境が変化する)
            if i % 50 == 0:
                current_concept_idx = (current_concept_idx + 1) % len(concepts)
                # 環境変化時はドーパミン（報酬/驚き）を与える
                # brain.reward(0.5)
                if hasattr(os_sys.brain, "motivation_system"):
                    os_sys.brain.motivation_system.update_state({"reward": 0.5})

            concept = concepts[current_concept_idx]

            # 入力生成: 概念 + ランダムノイズ
            # (同じ概念でも毎回微妙に異なるスパイクパターンになる)
            spikes = encoder(concept, duration=5)
            input_tensor = spikes.mean(dim=1) * 1.5

        elif phase == "sleep":
            # 睡眠中は外部入力なし（OS内部でリプレイが生成される）
            input_tensor = torch.zeros(1, 784).to(os_sys.device)

        # --- Run OS Cycle ---
        # observation = brain.run_cycle(input_tensor, phase=phase)
        observation = os_sys.run_cycle(input_tensor, phase=phase)

        # --- Data Collection ---
        # 必要なメトリクスを抽出
        record = {
            "cycle": i,
            "phase": 1 if phase == "wake" else 0,  # Plot用に数値化
            "energy": observation["bio_metrics"]["energy"],
            "fatigue": observation["bio_metrics"]["fatigue"],
            "dopamine": observation["bio_metrics"]["dopamine"],
            "synapse_count": observation["synapse_count"],
            "consciousness": observation["consciousness_level"],
            # 各領野の活性度
            "act_v1": observation["substrate_activity"].get("V1", 0),
            "act_assoc": observation["substrate_activity"].get("Association", 0),
            "act_motor": observation["substrate_activity"].get("Motor", 0),
            "memory_stored": observation.get("memory_stats", {}).get(
                "stored_episodes", 0
            ),
        }
        history.append(record)

        # プログレスバー更新
        pbar.set_description(f"Phase: {phase} | Energy: {record['energy']:.1f}")

    # 3. 保存
    output_dir = "runtime_state"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "experiment_history.json")

    with open(output_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"📄 Experiment data saved to {output_path}")
    return output_path


if __name__ == "__main__":
    run_experiment(cycles=1000)
