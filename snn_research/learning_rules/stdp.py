# ファイルパス: snn_research/learning_rules/stdp.py
# 日本語タイトル: STDP Learning Rule Module (Fixed)
# 目的・内容:
#   スパイクタイミング依存可塑性 (STDP) および報酬変調型STDP (R-STDP) の実装。
#   MyPyエラー修正: updateシグネチャの整合性確保、STDPエイリアスの追加。

from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor
from snn_research.learning_rules.base_rule import PlasticityRule


class STDPRule(PlasticityRule):
    """
    Spike-Timing Dependent Plasticity with Reward Modulation.
    
    設計方針書 4.1準拠:
    - 時間発展の性質を利用した学習
    - ドーパミンによる変調 (Reward-modulated)
    """

    def __init__(
        self,
        learning_rate: float = 0.005,
        tau_pre: float = 20.0,
        tau_post: float = 20.0,
        w_max: float = 5.0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.lr = learning_rate
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.w_max = w_max

    def update(
        self,
        pre_spikes: Tensor,
        post_spikes: Tensor,
        current_weights: Tensor,
        **kwargs: Any
    ) -> Tuple[Optional[Tensor], Dict[str, Any]]:
        """
        MyPy修正: 基底クラスに合わせてシグネチャを修正。
        local_state は kwargs から取得する。
        """
        local_state: Dict[str, Any] = kwargs.get("local_state", {})
        dt = kwargs.get("dt", 1.0)
        dopamine = kwargs.get("dopamine_level", 1.0) # デフォルトは等倍

        batch_size, in_features = pre_spikes.shape
        _, out_features = post_spikes.shape
        device = pre_spikes.device

        # --- トレースの初期化と更新 ---
        if "trace_pre" not in local_state:
            local_state["trace_pre"] = torch.zeros((batch_size, in_features), device=device)
        if "trace_post" not in local_state:
            local_state["trace_post"] = torch.zeros((batch_size, out_features), device=device)

        # トレース減衰
        decay_pre = torch.exp(torch.tensor(-dt / self.tau_pre))
        decay_post = torch.exp(torch.tensor(-dt / self.tau_post))
        
        # 新しいスパイクでトレース増加
        trace_pre = local_state["trace_pre"] * decay_pre + pre_spikes
        trace_post = local_state["trace_post"] * decay_post + post_spikes
        
        local_state["trace_pre"] = trace_pre
        local_state["trace_post"] = trace_post

        # --- STDP 更新則 ---
        # LTP: Pre(Trace) -> Post(Spike) : 因果関係あり (Preが先に発火)
        # LTD: Post(Trace) -> Pre(Spike) : 因果関係逆転 (Postが先に発火)
        
        # LTP: Postが発火した瞬間のPreトレース量に応じて強化
        # (batch, out, 1) * (batch, 1, in) -> (batch, out, in)
        delta_w_ltp = torch.matmul(post_spikes.unsqueeze(2), trace_pre.unsqueeze(1))
        
        # LTD: Preが発火した瞬間のPostトレース量に応じて減衰
        # (batch, out, 1) * (batch, 1, in) -> (batch, out, in)
        delta_w_ltd = torch.matmul(trace_post.unsqueeze(2), pre_spikes.unsqueeze(1))
        
        # 全体更新量 (バッチ平均)
        delta_w = (delta_w_ltp - delta_w_ltd).mean(dim=0)
        
        # ドーパミン変調 (R-STDP)
        # ドーパミンが多いと強化されやすく、少ないと変化が小さい
        delta_w = delta_w * self.lr * dopamine

        return delta_w, {"stdp_delta": delta_w.abs().mean().item()}

# Backward Compatibility Alias
STDP = STDPRule