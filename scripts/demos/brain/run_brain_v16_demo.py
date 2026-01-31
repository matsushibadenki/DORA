# ファイルパス: scripts/demos/brain/run_brain_v16_demo.py
# 日本語タイトル: Integrated Brain v16.4 Learning & Consciousness Demo (Fixed)
# 目的・内容:
#   - コンテナ初期化時に config.yaml をロードする処理を追加。

import sys
import os
import time
import logging
import torch
import numpy as np
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.containers import AppContainer
from snn_research.core.neuromorphic_os import NeuromorphicOS

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BrainDemo")

def generate_visual_stimulus(pattern_type: str = "random", device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    視覚野への入力刺激を生成する。
    SFormerのEmbedding層に合わせ、整数型のトークン列または特徴量を生成。
    """
    # 簡易的に [1, 128] のシーケンス（視覚トークン）を生成
    if pattern_type == "prey":
        # 「獲物」を表す特定のパターン（仮）
        return torch.randint(100, 200, (1, 128), device=device)
    elif pattern_type == "predator":
        # 「捕食者」を表す特定のパターン
        return torch.randint(800, 900, (1, 128), device=device)
    else:
        # ランダムノイズ
        return torch.randint(0, 1000, (1, 128), device=device)

def run_demo():
    print("\n" + "="*60)
    print("🧠 Neuromorphic OS & Artificial Brain v16.4 Integration Demo")
    print("="*60 + "\n")

    # 1. コンテナの初期化とシステムのブート
    container = AppContainer()
    
    # [Fix] 設定ファイルのロード (必須)
    config_path = Path("configs/templates/base_config.yaml")
    if not config_path.exists():
        # フォールバック: パスが見つからない場合、テスト用などでカレントが違う可能性考慮
        config_path = Path(__file__).resolve().parents[3] / "configs/templates/base_config.yaml"
    
    container.config.from_yaml(str(config_path))
    
    # 設定の上書き（デモ用）
    container.config.device.from_value("cpu") # 確実にCPUで動かす
    
    os_kernel: NeuromorphicOS = container.neuromorphic_os()
    brain = os_kernel.brain
    device = os_kernel.device

    print(f"✅ System Initialized on {device}")
    print(f"   - Brain Model: {type(brain).__name__}")
    print(f"   - OS Kernel: v1.1 (Tick: {os_kernel.tick_rate}Hz)")
    
    # OS起動
    os_kernel.boot()
    
    # 2. 学習サイクルの実行 (Wake Phase)
    print("\n🌞 [PHASE 1] WAKE CYCLE - Active Inference & Learning")
    
    episodes = [
        ("predator", "Run away!"),
        ("prey", "Chase it!"),
        ("random", "Ignore"),
        ("predator", "Run away!") # 再度提示して学習効果（反応速度など）を確認
    ]

    for i, (stimulus_type, expected_intent) in enumerate(episodes):
        print(f"\n⏱️  Episode {i+1}: Encountering '{stimulus_type}'")
        
        # 刺激生成
        visual_input = generate_visual_stimulus(stimulus_type, device)
        
        # OS経由でタスク投入（認知サイクル1ステップ）
        # 内部で: 知覚 -> 意識(Workspace) -> PFC/BG -> 行動
        start_time = time.time()
        result = os_kernel.submit_task(visual_input)
        process_time = time.time() - start_time
        
        # --- 結果の観察 ---
        
        # A. 意識の内容 (Conscious Broadcast)
        broadcast = result.get("conscious_broadcast", {})
        source_mod = broadcast.get("source", "None")
        print(f"   👁️  Consciousness: Focus on [{source_mod}]")
        
        # B. 動機・感情 (Drives)
        drives = result.get("drives", {})
        print(f"   ❤️  Internal State: Fear={drives.get('fear', 0.0):.2f}, Hunger={drives.get('hunger', 0.0):.2f}")
        
        # C. 意思決定 (Action)
        action = result.get("action")
        action_name = action['action'] if action else "No Action"
        print(f"   🤖 Action Selected: '{action_name}' (Confidence: {action.get('value', 0.0):.2f})")
        
        # D. PFCのゴール
        print(f"   🎯 PFC Goal: {result.get('pfc_goal')}")
        
        # 学習（可塑性）の確認
        # 報酬フィードバック（簡易実装）
        reward = 1.0 if stimulus_type == "prey" and action_name != "wait" else -0.1
        # 本来はTrainerクラスでbackwardするが、ここではBrain内部の状態更新を確認
        brain.motivation_system.update_state({"reward": reward})
        
        print(f"   ⚡ Processing Time: {process_time*1000:.1f}ms")
        time.sleep(0.5)

    # 3. 睡眠サイクルの実行 (Sleep Phase)
    print("\n🌙 [PHASE 2] SLEEP CYCLE - Memory Consolidation")
    
    # 意図的にエネルギーを下げて強制睡眠させるシナリオも可能だが、
    # ここではOSのコマンドで睡眠させる
    
    pre_sleep_stats = os_kernel.get_status_report()
    print(f"   🔋 Energy before sleep: {pre_sleep_stats['brain_status']['energy']:.1f}")
    
    os_kernel.shutdown() # Shutdown triggers sleep
    
    # 睡眠中の処理をシミュレート（実際は一瞬だが）
    time.sleep(1.0)
    print("   ... Dreaming & Consolidating Memories (Hippocampus -> Cortex) ...")
    
    # 再起動
    os_kernel.boot()
    
    post_sleep_stats = os_kernel.get_status_report()
    print(f"   🔋 Energy after sleep:  {post_sleep_stats['brain_status']['energy']:.1f}")
    print(f"   💤 Sleep Cycles Count:  {post_sleep_stats['brain_status']['cycle']}")

    # 4. 知識の確認 (RAG/Memory)
    print("\n📚 [PHASE 3] KNOWLEDGE CHECK")
    # 海馬にエピソードがたまっているか確認
    if hasattr(brain, 'hippocampus'):
        buffer_len = len(brain.hippocampus.episodic_buffer)
        print(f"   🧠 Hippocampus Buffer: {buffer_len} episodes stored.")
    
    # RAGに知識が転送されたか（モック動作ではあるが）
    knowledge = brain.retrieve_knowledge("predator")
    print(f"   📖 Retrieved Knowledge: {knowledge[:1]} ...")

    print("\n✅ Demo Completed Successfully.")

if __name__ == "__main__":
    run_demo()