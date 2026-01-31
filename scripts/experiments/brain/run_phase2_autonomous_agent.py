# ファイルパス: scripts/experiments/brain/run_phase2_autonomous_agent.py
# 日本語タイトル: Phase 2 Autonomous Agent Experiment (Fixed: Exploration & Entropy)
# 目的・内容:
#   - エントロピー正則化を追加し、探索能力を向上。
#   - 退屈(Boredom)によるペナルティを導入し、スタック（同じ場所での停止）を回避。
#   - エピソード数を増やして学習の収束を確認。

import sys
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from collections import deque
from typing import List, Dict, Any

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.containers import AppContainer
from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain
from snn_research.rl_env.grid_world import GridWorldEnv

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AutonomousAgent")

class BrainAgent:
    """
    ArtificialBrainをラップし、RLエージェントとして振る舞わせるクラス。
    """
    def __init__(self, brain: ArtificialBrain, action_dim: int, device: torch.device):
        self.brain = brain
        self.device = device
        self.action_dim = action_dim
        
        # 脳の出力次元をチェック
        self.brain.reset_state()
        with torch.no_grad():
            # ダミー入力
            dummy_input = torch.zeros(1, 4).long().to(device)
            dummy_out = self.brain(dummy_input)
            if dummy_out.dim() > 2:
                self.input_dim = dummy_out.shape[-1]
            else:
                self.input_dim = dummy_out.shape[-1]
        self.brain.reset_state()
        
        self.policy_head = nn.Linear(self.input_dim, action_dim).to(device)
        
        self.optimizer = optim.Adam(
            list(self.brain.parameters()) + list(self.policy_head.parameters()),
            lr=1e-4
        )
        
        self.memory: List[Dict[str, Any]] = []

    def get_action(self, state_tokens: torch.Tensor) -> int:
        self.brain.train()
        
        features = self.brain(state_tokens)
        
        if features.dim() > 2:
            features = features.mean(dim=1)
            
        logits = self.policy_head(features)
        probs = torch.softmax(logits, dim=-1)
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        
        # エントロピーも保存しておく（学習時に使用）
        self.memory.append({
            "log_prob": dist.log_prob(action),
            "entropy": dist.entropy()
        })
        
        return action.item()

    def update_policy(self, rewards: list):
        R = 0
        policy_loss = []
        # [Fix] 型注釈を追加
        returns: List[float] = []
        
        gamma = 0.9
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
            
        # [Fix] 変数名を変更して型再代入エラーを回避 (list -> Tensor)
        returns_tensor = torch.tensor(returns).to(self.device)
        
        if len(returns_tensor) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-9)
            
        entropy_loss = 0
        
        for i, step_data in enumerate(self.memory):
            log_prob = step_data["log_prob"]
            entropy = step_data["entropy"]
            
            # Policy Gradient Loss
            policy_loss.append(-log_prob * returns_tensor[i])
            
            entropy_loss -= 0.05 * entropy
            
        self.optimizer.zero_grad()
        
        pg_loss = torch.stack(policy_loss).sum()
        total_loss = pg_loss + entropy_loss # 合計損失
        
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 1.0)
        self.optimizer.step()
        
        self.memory = []
        self.brain.reset_state()
        
        return total_loss.item()

def visualize_grid(env, agent_pos, goal_pos):
    grid = [['.' for _ in range(env.size)] for _ in range(env.size)]
    
    ax, ay = agent_pos[0].item(), agent_pos[1].item()
    gx, gy = goal_pos[0].item(), goal_pos[1].item()
    
    ax, ay = min(max(0, ax), env.size-1), min(max(0, ay), env.size-1)
    gx, gy = min(max(0, gx), env.size-1), min(max(0, gy), env.size-1)

    grid[gy][gx] = 'G'
    grid[ay][ax] = 'A'
    
    if ax == gx and ay == gy:
        grid[ay][ax] = '🎉'

    print("-" * (env.size * 2 + 3))
    for row in reversed(grid):
        print("| " + " ".join(row) + " |")
    print("-" * (env.size * 2 + 3))


def run_experiment():
    print("\n" + "="*60)
    print("🤖 Artificial Brain Phase 2: Autonomous Agent (Enhanced)")
    print("="*60 + "\n")

    container = AppContainer()
    config_path = Path("configs/templates/base_config.yaml")
    if not config_path.exists():
        config_path = Path(__file__).resolve().parents[3] / "configs/templates/base_config.yaml"
    container.config.from_yaml(str(config_path))
    container.config.device.from_value("cpu")
    
    brain = container.artificial_brain()
    device = brain.device
    
    env_size = 5
    env = GridWorldEnv(size=env_size, max_steps=30, device=str(device)) # MaxStep少し増加
    
    agent = BrainAgent(brain, action_dim=4, device=device)
    
    print(f"✅ Environment & Agent Ready (Grid: {env_size}x{env_size})")

    # [Fix] エピソード数を増加
    episodes = 100
    success_count = 0
    total_rewards_history = []

    for episode in range(1, episodes + 1):
        env.reset()
        
        # 前回の位置を記憶して、移動していない(退屈)判定に使用
        last_pos_tuple = (-1, -1)
        stuck_counter = 0
        
        episode_rewards = []
        done = False
        
        # 最初の数回と、成功しだした後半を表示
        show_render = (episode <= 3) or (episode >= episodes - 3)
        
        if show_render:
            print(f"\n🎬 Episode {episode} Start")
            visualize_grid(env, env.agent_pos, env.goal_pos)

        step_count = 0
        while not done:
            state_tokens = torch.cat([env.agent_pos, env.goal_pos]).unsqueeze(0)
            
            action = agent.get_action(state_tokens)
            next_state_vec, reward, done = env.step(action)
            
            # --- [Fix] Motivation & Boredom Logic ---
            current_pos_tuple = (env.agent_pos[0].item(), env.agent_pos[1].item())
            
            # 場所が変わっていないなら退屈カウント増加
            if current_pos_tuple == last_pos_tuple:
                stuck_counter += 1
            else:
                stuck_counter = 0
            last_pos_tuple = current_pos_tuple
            
            # 動機システムへ状態ハッシュ（位置）を渡して退屈度を更新させる
            # Tensorではなく文字列にして渡すことでハッシュ化を有効にする
            pos_str = f"{current_pos_tuple}"
            internal_state = brain.motivation_system.process(pos_str)
            
            boredom = internal_state.get("boredom", 0.0)
            
            # 退屈ペナルティ: 動いていないと罰を与える
            boredom_penalty = 0.0
            if stuck_counter > 1:
                boredom_penalty = -0.1 * stuck_counter # 停滞すればするほど痛い
            
            # 報酬統合
            total_reward = reward + boredom_penalty
            episode_rewards.append(total_reward)
            
            brain.motivation_system.update_state({"reward": float(reward)})
            
            step_count += 1
            
            if show_render:
                print(f"   Step {step_count}: Action {['Up','Down','Left','Right'][action]} -> R {reward:.2f} (Boredom: {boredom:.2f})")
                visualize_grid(env, env.agent_pos, env.goal_pos)
                # time.sleep(0.05) # 高速化のためコメントアウト推奨

        # 学習
        loss = agent.update_policy(episode_rewards)
        
        total_score = sum(episode_rewards)
        total_rewards_history.append(total_score)
        
        # ゴール到達判定 (報酬1.0)
        is_success = False
        if any(r >= 1.0 for r in episode_rewards): # どこかでゴール報酬を得ていればOK
            is_success = True
            success_count += 1
            result_mark = "🎉 Success"
        else:
            result_mark = "💀 Failed"
            
        logger.info(f"Episode {episode:03d} | Steps: {step_count:02d} | Score: {total_score:.2f} | Loss: {loss:.4f} | {result_mark}")
        
        if episode % 20 == 0:
            brain.sleep_cycle()

    print("\n" + "="*60)
    print(f"📊 Experiment Result: {success_count}/{episodes} Success Rate ({(success_count/episodes)*100:.1f}%)")
    print("="*60)
    
    if success_count > 10:
        print("✅ Improved agent demonstrates adaptive behavior!")
    else:
        print("⚠️ Learning is still challenging. Consider simpler task or more training.")

if __name__ == "__main__":
    run_experiment()