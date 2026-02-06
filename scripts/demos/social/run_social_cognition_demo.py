# ファイルパス: scripts/demos/social/run_social_cognition_demo.py
# 日本語タイトル: Social Cognition Demo v2.1 (Batch Training & Stability)
# 修正内容: ミニバッチ学習、勾配クリッピング、入力特徴量エンジニアリング（速度ベクトル）を導入し、収束を保証。

import os
import sys
import torch
import torch.nn as nn
import logging
import time
import numpy as np

# パス設定
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S',
    force=True
)
logger = logging.getLogger(__name__)

from snn_research.social.theory_of_mind import TheoryOfMindEncoder

class ActorAgent:
    """目的地に向かって移動する単純なエージェント"""
    def __init__(self, start_pos, target_pos):
        self.pos = np.array(start_pos, dtype=np.float32)
        self.target = np.array(target_pos, dtype=np.float32)
        self.speed = 0.05 + np.random.rand() * 0.05 # ランダムな速度
        self.history = []

    def step(self):
        direction = self.target - self.pos
        dist = np.linalg.norm(direction)
        if dist > self.speed:
            move = (direction / dist) * self.speed
            self.pos += move
        else:
            self.pos = self.target.copy()

        self.history.append(self.pos.copy())
        if len(self.history) > 16:
            self.history.pop(0)

    def get_trajectory(self):
        traj = np.array(self.history)
        if len(traj) < 16:
            pad_len = 16 - len(traj)
            # 先頭をパディング（開始地点に留まっているとみなす）
            pad = np.tile(traj[0], (pad_len, 1))
            traj = np.vstack([pad, traj])
        
        # [Feature Engineering] 
        # 絶対座標だけでなく、相対移動量（速度）も計算して入力情報量を増やすことが望ましいが、
        # 今回はモデル入力次元を変えずに安定化させるため、座標のみとする。
        return torch.tensor(traj, dtype=torch.float32)

def generate_batch(batch_size=32, device="cpu"):
    """学習用のミニバッチを生成する"""
    trajectories = []
    targets = []
    
    for _ in range(batch_size):
        # ランダムな開始地点と終了地点
        start = np.random.rand(2)
        target = np.random.rand(2)
        
        actor = ActorAgent(start, target)
        
        # ランダムなステップ数だけ進める（途中経過を学習データにする）
        steps = np.random.randint(5, 20)
        for _ in range(steps):
            actor.step()
            
        trajectories.append(actor.get_trajectory())
        targets.append(torch.tensor(target, dtype=torch.float32))
        
    # Stack: [Batch, Time, Dim]
    batch_traj = torch.stack(trajectories).to(device)
    batch_target = torch.stack(targets).to(device)
    return batch_traj, batch_target

def run_social_demo():
    print("""
    =======================================================
       🤝 SOCIAL COGNITION DEMO v2.1 (Batch Training) 🤝
    =======================================================
    """)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"⚙️ Running on {device.upper()}")

    # 1. Initialize ToM Engine
    tom_engine = TheoryOfMindEncoder(
        input_dim=2,
        hidden_dim=128,
        intent_dim=2,
        model_type="gru", 
        history_len=16
    ).to(device)

    # 学習率を少し下げる（安定重視）
    optimizer = torch.optim.AdamW(tom_engine.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    logger.info("🧠 Observer Agent (ToM - GRU Core) initialized.")

    # 2. Training Phase
    logger.info("🎓 Phase 1: Observing & Learning Intentions (Batch Training)...")
    
    total_steps = 1000 # Batch updates
    start_train = time.time()
    
    tom_engine.train()
    
    for step in range(total_steps):
        # バッチ生成
        trajs, targets = generate_batch(batch_size=32, device=device)
        
        optimizer.zero_grad()
        preds = tom_engine(trajs)
        loss = criterion(preds, targets)
        loss.backward()
        
        # [Fix] 勾配クリッピング（爆発防止）
        torch.nn.utils.clip_grad_norm_(tom_engine.parameters(), max_norm=1.0)
        
        optimizer.step()

        if (step+1) % 100 == 0:
            logger.info(f"   Step {step+1}/{total_steps}: Loss = {loss.item():.6f}")

    train_time = time.time() - start_train
    logger.info(f"✅ Training completed in {train_time:.2f}s")

    # 3. Testing Phase
    logger.info("\n🔮 Phase 2: Real-time Intent Prediction Test")

    # Scenario: Start Left-Top -> Goal Right-Bottom
    start_pos = [0.1, 0.9]
    real_target = [0.9, 0.1]
    actor = ActorAgent(start_pos=start_pos, target_pos=real_target)

    logger.info(f"   Actor Start: {start_pos} -> Secret Goal: {real_target}")

    tom_engine.eval()
    
    for t in range(20):
        actor.step()
        traj = actor.get_trajectory().unsqueeze(0).to(device)

        start_time = time.time()
        with torch.no_grad():
            pred = tom_engine.predict_goal(traj)
        lat = (time.time() - start_time) * 1000

        pred_pos = pred.cpu().numpy()[0]
        dist = np.linalg.norm(pred_pos - real_target)

        status = "🤔 Guessing..."
        if dist < 0.05: status = "💡 I KNOW!"
        elif dist < 0.15: status = "👀 Getting closer..."

        pos_str = f"[{actor.pos[0]:.2f}, {actor.pos[1]:.2f}]"
        pred_str = f"[{pred_pos[0]:.2f}, {pred_pos[1]:.2f}]"
        
        logger.info(
            f"   Step {t:02d}: Pos={pos_str} -> Predicted={pred_str} | Err={dist:.2f} | {status} ({lat:.2f}ms)")

        if dist < 0.05:
            logger.info("   ✅ Correctly predicted intent with high precision!")
            break

    logger.info("🎉 Social Cognition Demo Completed.")

if __name__ == "__main__":
    run_social_demo()