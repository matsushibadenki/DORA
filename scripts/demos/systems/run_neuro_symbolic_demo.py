# scripts/demos/systems/run_neuro_symbolic_demo.py
import sys
import os
import logging
import torch
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

try:
    from snn_research.cognitive_architecture.neuro_symbolic_bridge import NeuroSymbolicBridge
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', force=True)
logger = logging.getLogger("NeuroSymbolicDemo")

def run_demo():
    logger.info("🌉 Initializing Neuro-Symbolic Bridge Demo...")
    
    concepts = ["Apple", "Fire truck", "Sun", "Rose"]
    
    # 1. コンポーネントの準備 (引数名を修正)
    bridge = NeuroSymbolicBridge(
        input_dim=128,   # snn_output_dim -> input_dim
        embed_dim=512,   # symbol_dim -> embed_dim
        concepts=concepts
    )
    # device引数は__init__にないので、後からto()で指定
    device = torch.device("cpu")
    bridge.to(device)
    
    # 2. SNNからの信号シミュレーション (直感的な出力)
    # 例: 何か「赤い丸いもの」を見たときのスパイク発火パターン (Batch, Dim)
    snn_signal = torch.rand(1, 128).to(device)
    logger.info(f"🧠 SNN Signal (Intuition) Received. Shape: {snn_signal.shape}")
    
    # 3. シンボリック推論への変換 (Grounding/Extraction)
    # bridge.ground() は存在しないため、extract_symbols() を使用
    logger.info("🔄 Extracting Symbols from SNN signal...")
    detected_symbols = bridge.extract_symbols(snn_signal, threshold=0.3)
    
    if detected_symbols:
        logger.info(f"💡 Detected Concepts: {[s.name for s in detected_symbols]}")
    else:
        logger.info("💡 No clear concept detected (Simulating ambiguity).")
    
    # 4. 逆方向: シンボルからSNN信号への変換 (Modulation/Injection)
    # bridge.modulate() は存在しないため、symbol_to_spike() を使用
    target_concept = "Apple"
    logger.info(f"↩️ Injecting Top-down Attention for '{target_concept}'...")
    
    # 文字列からテンソルへ変換
    feedback_signal = bridge.symbol_to_spike(target_concept, batch_size=1)
    
    logger.info(f"✅ Feedback Signal Generated: {feedback_signal.shape}")
    logger.info("   (This signal acts as an attractor bias for the SNN)")
    
    logger.info("🎉 Neuro-Symbolic Cycle Complete.")

if __name__ == "__main__":
    run_demo()