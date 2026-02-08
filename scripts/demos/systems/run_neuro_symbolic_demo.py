# directory: scripts/demos/systems
# file: run_neuro_symbolic_demo.py
# purpose: Neuro-Symbolic連携デモ (Fix: Double Normalization Bug)

import torch
from torchvision import datasets, transforms
import sys
import os
import random
import numpy as np

# パス設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

try:
    from snn_research.models.adapters.sara_adapter import SARAAdapter
    from snn_research.cognitive_architecture.neuro_symbolic_bridge import NeuroSymbolicBridge
except ImportError:
    print("Error: Required modules not found. Please ensure project structure is correct.")
    sys.exit(1)

def add_noise(img, noise_level=0.3):
    """
    画像にノイズを加える
    img: [0, 1] のTensor
    """
    noise = torch.randn_like(img) * noise_level
    # Clampして [0, 1] の範囲に収める (Adapterが後で正規化するため)
    return (img + noise).clamp(0, 1)

def run_demo():
    print("=== DORA Neuro-Symbolic Integration Demo (Fixed) ===")
    
    model_path = "models/checkpoints/sara_mnist_v5.pth"
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Please run training first.")
        return

    # エージェント初期化
    sara = SARAAdapter(model_path, device="cpu")
    
    # ブリッジ初期化
    bridge = NeuroSymbolicBridge(sara, confidence_threshold=0.85)
    
    # データセット (正規化なしのRaw Tensorとしてロード)
    dataset = datasets.MNIST('../data', train=False, download=True, 
                           transform=transforms.ToTensor())
    
    print("Starting interaction loop...")
    print("-" * 60)
    
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    for step in range(1, 11):
        idx = indices[step]
        img_clean, label = dataset[idx] # [0, 1] Tensor
        
        # 難問生成（ノイズ付加）
        is_hard = (step % 3 == 0)
        if is_hard:
            # SARAを迷わせるレベルのノイズ
            img_input = add_noise(img_clean, noise_level=0.8)
            difficulty = "HARD (Noisy)"
        else:
            img_input = img_clean
            difficulty = "EASY"
            
        print(f"\nStep {step}: Input [{label}] - Difficulty: {difficulty}")
        
        # ブリッジ処理
        result = bridge.process_input(img_input, correct_label=label)
        
        # 結果表示
        source = result["source"]
        pred = result["prediction"]
        conf = result["confidence"]
        latency = result["latency"]
        note = result.get("explanation", "")
        
        status = "✅" if pred == label else "❌"
        
        print(f"  -> Processed by: {source}")
        print(f"  -> Result: {pred} {status} (Conf: {conf:.1%})")
        print(f"  -> Latency: {latency:.1f}ms | Note: {note}")
        
        # 学習後の確認
        if "System 2" in source and "learning_loss" in result:
            print(f"  [Verification] Re-evaluating with System 1...")
            retry = bridge.process_input(img_input) # Teacherなし
            
            r_pred = retry["prediction"]
            r_conf = retry["confidence"]
            r_src = retry["source"]
            
            # System 1が自信を持って正解できれば成功
            if r_pred == label and r_src.startswith("System 1") and r_conf > 0.8:
                print(f"  ✨ SUCCESS: System 1 learned to handle this noise pattern! (Conf: {r_conf:.1%})")
            else:
                print(f"  ⚠️  PARTIAL: Adaptation incomplete. (Pred: {r_pred}, Conf: {r_conf:.1%})")
            
    bridge.print_stats()

if __name__ == "__main__":
    run_demo()