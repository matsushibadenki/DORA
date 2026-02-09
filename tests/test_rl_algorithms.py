# directory: tests
# file: test_rl_algorithms.py
# title: Test RL Algorithms
# purpose: 強化学習アルゴリズム（PPOなど）の基本動作確認

import torch
import pytest
from snn_research.training.rl.spike_ppo import SpikePPO

def test_ppo_step():
    """
    SpikePPOの基本ステップ実行テスト
    """
    state_dim = 4
    action_dim = 2
    
    # 修正箇所: 引数名を is_continuous から continuous_action に変更
    # SARAバックエンドがなくても動作するよう use_sara_backend=False も明示的にテストに含める（任意）
    agent = SpikePPO(
        state_dim=state_dim, 
        action_dim=action_dim, 
        continuous_action=True,  # ここを修正
        use_sara_backend=False   # テスト環境によってはSARAがない場合もあるためLegacyモードで通す
    )
    
    state = torch.randn(state_dim)
    
    # アクション選択のテスト
    action = agent.select_action(state)
    
    # 連続値アクションの確認
    if isinstance(action, (list, tuple)):
        # 配列で返ってくる場合
        assert len(action) == action_dim
    else:
        # Numpy配列の場合
        assert action.shape == (action_dim,) or action.size == action_dim

def test_ppo_update():
    """
    SpikePPOの学習更新ステップのテスト
    """
    state_dim = 4
    action_dim = 2
    agent = SpikePPO(state_dim, action_dim, continuous_action=True, use_sara_backend=False)
    
    # ダミーデータの蓄積
    for _ in range(10):
        state = torch.randn(state_dim)
        action = agent.select_action(state)
        reward = 1.0
        done = False
        agent.store_reward(reward, done)
    
    # updateメソッドがエラーなく走るか確認
    try:
        agent.update()
    except Exception as e:
        pytest.fail(f"PPO update failed with error: {e}")

if __name__ == "__main__":
    test_ppo_step()
    test_ppo_update()
    print("All RL tests passed!")