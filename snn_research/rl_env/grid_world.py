# ファイルパス: snn_research/rl_env/grid_world.py
# 日本語タイトル: Grid World Environment (Reward Shaping)
# 目的・内容:
#   - ゴールまでのマンハッタン距離に基づき、近づいたら報酬、遠ざかったら罰を与えるように変更。
#   - これにより学習の立ち上がりを劇的に改善する。

import torch
from typing import Tuple

class GridWorldEnv:
    def __init__(self, size: int = 5, max_steps: int = 50, device: str = 'cpu'):
        self.size = size
        self.max_steps = max_steps
        self.device = device
        
        self.agent_pos = torch.zeros(2, device=self.device, dtype=torch.long)
        self.goal_pos = torch.zeros(2, device=self.device, dtype=torch.long)
        
        self.current_step = 0
        self.reset()

    def _get_state(self) -> torch.Tensor:
        state = torch.cat([
            (self.agent_pos / (self.size - 1)) * 2 - 1,
            (self.goal_pos / (self.size - 1)) * 2 - 1
        ]).float()
        return state

    def reset(self) -> torch.Tensor:
        self.agent_pos = torch.randint(0, self.size, (2,), device=self.device)
        self.goal_pos = torch.randint(0, self.size, (2,), device=self.device)
        while torch.equal(self.agent_pos, self.goal_pos):
            self.goal_pos = torch.randint(0, self.size, (2,), device=self.device)
        
        self.current_step = 0
        # 前回の距離をリセット
        self.prev_distance = self._calculate_distance()
        return self._get_state()

    def _calculate_distance(self) -> float:
        # マンハッタン距離
        return float(torch.sum(torch.abs(self.agent_pos - self.goal_pos)).item())

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        self.current_step += 1

        if action == 0: # 上
            self.agent_pos[1] += 1
        elif action == 1: # 下
            self.agent_pos[1] -= 1
        elif action == 2: # 左
            self.agent_pos[0] -= 1
        elif action == 3: # 右
            self.agent_pos[0] += 1
        
        # 壁にぶつかったかどうかの判定 (範囲外に出ようとしたら位置は変わらない)
        clamped_pos = torch.clamp(self.agent_pos, 0, self.size - 1)
        hit_wall = not torch.equal(self.agent_pos, clamped_pos)
        self.agent_pos = clamped_pos

        current_distance = self._calculate_distance()
        
        reward = -0.01  # 基本ステップコスト
        done = False

        if torch.equal(self.agent_pos, self.goal_pos):
            reward = 1.0  # ゴール到達
            done = True
        elif hit_wall:
            reward = -0.1 # 壁衝突ペナルティ
        else:
            # 距離ベースのシェイピング報酬
            # 近づいたら +0.1, 遠ざかったら -0.1
            diff = self.prev_distance - current_distance
            if diff > 0:
                reward += 0.1
            elif diff < 0:
                reward += -0.1
        
        self.prev_distance = current_distance

        if self.current_step >= self.max_steps:
            done = True
        
        next_state = self._get_state()

        return next_state, reward, done