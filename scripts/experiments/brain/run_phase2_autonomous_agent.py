# scripts/experiments/brain/run_phase2_autonomous_agent.py
import sys
import os
import logging
import torch
import torch.nn as nn
import numpy as np
import time

# プロジェクトルート
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', force=True)
logger = logging.getLogger("Phase2_Autonomous_Agent")

class BrainAgent:
    """
    Phase 2 Brainを搭載した自律エージェント。
    環境からの入力を処理し、行動を決定する。
    """
    def __init__(self, brain, action_dim=4, device="cpu", input_dim=128): # input_dim引数を追加
        self.brain = brain
        self.device = device
        self.action_head = nn.Linear(64, action_dim).to(device) # 仮の出力層
        self.input_dim = input_dim
        
        # ダミー実行でシェイプ確認
        # ArtificialBrainが期待する次元に合わせてダミー入力を生成
        # エラーログから判断すると (Batch, InputDim) の2次元入力を期待している可能性が高い
        dummy_input = torch.randn(1, self.input_dim).to(device)
        
        try:
            _ = self.brain(dummy_input)
            logger.info("✅ Brain forward pass check successful.")
        except Exception as e:
            logger.warning(f"⚠️ Brain forward pass check failed (might be ok if lazy init): {e}")

    def get_action(self, observation):
        """
        観測 -> 脳 -> 行動
        """
        self.brain.eval()
        with torch.no_grad():
            # 観測データの前処理
            if isinstance(observation, np.ndarray):
                obs_tensor = torch.from_numpy(observation).float().to(self.device)
            elif isinstance(observation, torch.Tensor):
                obs_tensor = observation.float().to(self.device)
            else:
                obs_tensor = torch.tensor(observation).float().to(self.device)
            
            # 入力次元の調整 (Batch次元の追加)
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            
            # 脳による処理
            brain_output = self.brain(obs_tensor)
            
            # 行動決定 (仮: 脳の出力の一部を行動とみなす)
            if isinstance(brain_output, dict):
                # 辞書の場合は値リストの最初の要素を取得（実装依存）
                feat = list(brain_output.values())[0]
            else:
                feat = brain_output

            if isinstance(feat, torch.Tensor):
                # 次元合わせ
                if feat.shape[-1] != 64:
                     # 簡易的な射影 (デモ用)
                     proj = nn.Linear(feat.shape[-1], 64).to(self.device)
                     feat = proj(feat)
                
                action_logits = self.action_head(feat)
                action = torch.argmax(action_logits, dim=-1).item()
                return action
            
            return 0 # Fallback

def run_experiment():
    print("\n============================================================")
    print("   Artificial Brain Phase 2: Autonomous Agent (Enhanced)")
    print("============================================================")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 脳の構成設定
    # エラーメッセージより、ArtificialBrainの内部モデルは input_dim=128 を期待している
    input_dim = 128 
    
    config = {
        "input_dim": input_dim, 
        "hidden_dim": 512, # エラーメッセージの weight shape (128x512) から推測
        "output_dim": 64
    }
    
    # ArtificialBrainの初期化
    # config辞書を渡す形式か、kwargsで渡す形式か実装によるが、ここでは両方に対応できるよう配慮
    try:
        brain = ArtificialBrain(config)
    except:
        brain = ArtificialBrain(input_dim=input_dim, hidden_dim=512)
        
    brain.to(device)
    
    # エージェント化
    agent = BrainAgent(brain, action_dim=4, device=device, input_dim=input_dim)
    
    logger.info("🤖 Agent initialized. Starting autonomous loop...")
    
    # 自律ループ
    try:
        for step in range(10):
            # ダミー環境からの観測 (入力次元を合わせる)
            observation = np.random.randn(input_dim).astype(np.float32) 
            
            logger.info(f"--- Step {step+1} ---")
            
            action = agent.get_action(observation)
            
            actions_map = {0: "Move Forward", 1: "Turn Left", 2: "Turn Right", 3: "Interact"}
            action_str = actions_map.get(action, "Idle")
            
            logger.info(f"🧠 Brain Decision: {action_str} (ID: {action})")
            
            time.sleep(0.5)
            
        logger.info("✅ Autonomous Agent Loop Finished Successfully.")
        
    except KeyboardInterrupt:
        logger.info("🛑 Agent stopped by user.")
    except Exception as e:
        logger.error(f"❌ Critical Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_experiment()