# ファイルパス: scripts/demos/learning/run_sleep_cycle_demo.py
# Title: Autonomous Sleep Cycle Demo (Fixed v2)
# Description:
#   日中の活動で記憶を蓄積し、疲労後に睡眠をとって記憶を長期記憶へ転送するデモ。
#   [Fix] AstrocyteNetwork.consume_energy の引数を修正。
#   [Fix] SFormerモデルに合わせて入力形式を整数(Token IDs)に変更。
#   [Fix] SDFT (自己蒸留) 用のフックを追加。

import sys
import os
import torch
import time
import logging

# パス設定
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../")))

# 循環参照回避のため、必要なクラスのみインポート
from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.hippocampus import Hippocampus
from snn_research.cognitive_architecture.cortex import Cortex

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    force=True
)
logger = logging.getLogger("SleepCycleDemo")

def run_sleep_cycle_demo():
    print("=== 🌙 Autonomous Sleep Cycle Demo ===")
    print("日中の活動で記憶を蓄積し、疲労後に睡眠をとって記憶を長期記憶へ転送します。\n")

    # 1. コンポーネントの初期化
    workspace = GlobalWorkspace(dim=64)
    
    # Astrocyte (エネルギー管理)
    astrocyte = AstrocyteNetwork(initial_energy=1000.0, max_energy=1000.0)
    
    # 長期記憶 (Cortex)
    cortex = Cortex()
    
    # 海馬 (短期記憶) - 容量を小さくして溢れさせるシミュレーション
    hippocampus = Hippocampus(short_term_capacity=5, working_memory_dim=64)
    
    # 脳の構成設定
    brain_config = {
        "model": {
            "d_model": 64, # モデルの内部次元
            "vocab_size": 1000 # 語彙サイズ
        },
        "device": "cpu" # 強制的にCPUまたはMPSを使うよう指定
    }

    # 脳の構築 (依存性の注入)
    try:
        brain = ArtificialBrain(
            config=brain_config,
            global_workspace=workspace,
            astrocyte_network=astrocyte,
            hippocampus=hippocampus,
            cortex=cortex
        )
    except TypeError as e:
        print(f"❌ Initialization Error: {e}")
        print("Tip: ArtificialBrain.__init__ arguments might need adjustment.")
        return

    # 2. 日中の活動 (Learning Phase)
    print("☀️ Day 1: Learning & Exploration Started")
    
    experiences = [
        "Saw a red apple on the table.",
        "Heard a loud noise from the street.",
        "Read a book about neural networks.",
        "Felt tired after coding python.",
        "Ate a delicious sandwich."
    ]

    for i, exp in enumerate(experiences):
        # [Fix] Transformerモデルへの入力は整数ID (Token IDs) である必要がある
        # (Batch=1, SeqLen=10, 値は0-999のランダム)
        sensory_input_ids = torch.randint(0, 1000, (1, 10)).to(brain.device)
        
        # 脳活動 (Forward)
        brain.process_step(sensory_input_ids)
        
        # 海馬へエピソード記憶を保存
        # processメソッド内でのembedding処理用にテキストを渡す
        memory_item = {
            "embedding": torch.randn(1, 64), # 埋め込みは別途計算される想定(簡易的にランダム)
            "text": exp,
            "timestamp": time.time(),
            # SDFT (自己蒸留) 用のデータ形式
            "input": exp,
            "answer": "Experience Log"
        }
        brain.hippocampus.process(memory_item)
        
        # [Fix] SDFT連携: SleepConsolidatorにもエピソードを通知 (もし存在すれば)
        if hasattr(brain, "sleep_manager") and brain.sleep_manager:
             brain.sleep_manager.add_episode(memory_item)
        
        # [Fix] エネルギー消費 (引数はamountのみ)
        # amount=15.0
        brain.astrocyte.consume_energy(15.0)
        
        print(f"  Step {i+1}: Experiencing -> '{exp}'")
        time.sleep(0.1)

    # バッファ確認
    buffer_len = len(brain.hippocampus.episodic_buffer)
    # sleep_manager側のバッファも確認
    sm_buffer_len = len(brain.sleep_manager.episodic_buffer) if hasattr(brain, "sleep_manager") else 0
    
    print(f"\n🧠 Hippocampus Buffer: {buffer_len} items")
    print(f"🧠 Sleep Manager Buffer: {sm_buffer_len} items (Ready for dreaming)")
    
    energy_level = brain.astrocyte.get_energy_level()
    print(f"⚡ Current Energy: {energy_level:.1f}/1000")

    # 3. 疲労と睡眠の必要性 (Fatigue Phase)
    print("\n😫 Energy dropped critically low. Needing sleep...")
    brain.astrocyte.energy = 10.0 # 強制的に枯渇させる
    print(f"   (Energy forced down to: {brain.astrocyte.energy})")

    # 4. 睡眠サイクル (Sleep Phase)
    print("\n🌙 Processing next step (Checking for sleep need)...")
    
    # ダミー入力
    dummy_input = torch.randint(0, 1000, (1, 10)).to(brain.device)
    result = brain.process_step(dummy_input)
    
    # 睡眠トリガー条件の確認
    should_sleep = (
        result.get("status") == "exhausted" or 
        brain.astrocyte.get_energy_level() < 20.0
    )
    
    if should_sleep:
        print("💤 Brain triggered SLEEP MODE due to exhaustion.")
        
        # 睡眠実行 (エネルギー回復 & 夢/SDFT)
        # perform_sleep_cycleを明示的に呼び出す
        sleep_report = brain.perform_sleep_cycle(cycles=3)
        print(f"   > Sleep Report: {sleep_report}")
        
        if sleep_report.get("learned_samples", 0) > 0:
            print("   🦄 Dreaming (SDFT) happened! New knowledge distilled.")
        
        # 記憶の固定化 (Hippocampus -> Cortex)
        print("   > Consolidating memories from Hippocampus to Cortex...")
        memories = brain.hippocampus.flush_memories()
        transferred_count = len(memories)
        
        print(f"   > Memories Transferred: {transferred_count}")
        
        print("✨ Woke up refreshed!")
        print(f"⚡ Energy recovered: {brain.astrocyte.energy:.1f}")
    else:
        print("❌ Sleep was not triggered. Logic check needed.")
        print(f"Debug Result: {result}")
        print(f"Debug Energy: {brain.astrocyte.get_energy_level()}")

    # 5. 結果確認 (Evaluation)
    print("\n📚 Checking Result...")
    
    if 'transferred_count' in locals() and transferred_count > 0:
        print("\n✅ SUCCESS: Sleep cycle completed, dreams simulated, and memories consolidated.")
    else:
        print("\n⚠️ PARTIAL SUCCESS: Sleep logic ran, but memory transfer count is 0.")

    print("\n=== Demo Finished ===")

if __name__ == "__main__":
    run_sleep_cycle_demo()