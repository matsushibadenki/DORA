# ファイルパス: snn_research/learning_rules/stdp.py
# 日本語タイトル: STDP (Spike-Timing-Dependent Plasticity) Rule
# 目的・内容:
#   生物学的妥当性の高い局所学習則の実装。
#   前シナプスと後シナプスのスパイクタイミング差に基づいて重みを強化(LTP)・弱化(LTD)する。
#   海馬などの短期記憶・連想記憶モジュールで使用される。

import torch
import logging
from typing import Dict, Any, Tuple, Optional

from snn_research.learning_rules.base_rule import PlasticityRule

logger = logging.getLogger(__name__)

class STDPRule(PlasticityRule):
    """
    Spike-Timing-Dependent Plasticity (STDP) Rule.
    
    dw = A_plus * pre_trace * post_spike  (LTP: Pre -> Post)
       - A_minus * post_trace * pre_spike (LTD: Post -> Pre)
       
    Args:
        learning_rate (float): 基本学習率
        tau_pre (float): 前シナプストレースの時定数 (ms)
        tau_post (float): 後シナプストレースの時定数 (ms)
        w_max (float): 重みの最大値（クリッピング用）
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        tau_pre: float = 20.0,
        tau_post: float = 20.0,
        w_max: float = 1.0
    ):
        self.lr = learning_rate
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.w_max = w_max
        
        # LTP/LTDのバランス比率
        self.A_plus = 1.0
        self.A_minus = 1.05 # LTDをわずかに強くして発火暴走を防ぐ

    def update(
        self, 
        pre_spikes: torch.Tensor, 
        post_spikes: torch.Tensor, 
        current_weights: torch.Tensor, 
        local_state: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """
        STDPによる重み更新計算。
        
        Args:
            pre_spikes (Tensor): (Batch, N_pre)
            post_spikes (Tensor): (Batch, N_post)
            current_weights (Tensor): (N_post, N_pre)
            local_state (Dict): 前回のトレース状態を保持するための辞書
        """
        if local_state is None:
            local_state = {}

        batch_size = pre_spikes.shape[0]
        dt = kwargs.get("dt", 1.0) # タイムステップ幅

        # --- 1. Trace Update ---
        # トレース: スパイクが発生すると1にジャンプし、指数関数的に減衰する値
        
        # Pre-synaptic trace
        trace_pre = local_state.get("trace_pre", torch.zeros_like(pre_spikes))
        decay_pre = 1.0 - (dt / self.tau_pre)
        trace_pre = trace_pre * decay_pre + pre_spikes
        
        # Post-synaptic trace
        trace_post = local_state.get("trace_post", torch.zeros_like(post_spikes))
        decay_post = 1.0 - (dt / self.tau_post)
        trace_post = trace_post * decay_post + post_spikes

        # 状態の保存
        local_state["trace_pre"] = trace_pre.detach()
        local_state["trace_post"] = trace_post.detach()

        # --- 2. Weight Update Calculation ---
        # LTP: Pre Trace + Post Spike (Preが先に発火し、その影響が残っている間にPostが発火)
        # (Batch, N_post, 1) * (Batch, 1, N_pre) -> (Batch, N_post, N_pre)
        # post_spikesを使うタイミングでLTPが発生
        delta_w_ltp = torch.einsum("bi,bj->bij", post_spikes, trace_pre) * self.A_plus

        # LTD: Post Trace + Pre Spike (Postが先に発火し、その後Preが発火 -> 因果逆転)
        # pre_spikesを使うタイミングでLTDが発生
        delta_w_ltd = torch.einsum("bi,bj->bij", trace_post, pre_spikes) * self.A_minus
        
        # 統合
        delta_w = (delta_w_ltp - delta_w_ltd).mean(dim=0) * self.lr

        # 重み安定化のためのソフトバウンド (Optional)
        # wが大きくなるとLTPが弱まり、小さくなるとLTDが弱まる
        # delta_w = delta_w_ltp * (self.w_max - current_weights) - delta_w_ltd * (current_weights)

        logs = {
            "mean_ltp": delta_w_ltp.mean().item(),
            "mean_ltd": delta_w_ltd.mean().item(),
            "max_trace_pre": trace_pre.max().item()
        }

        return delta_w, logs

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "STDP",
            "lr": self.lr,
            "tau_pre": self.tau_pre,
            "tau_post": self.tau_post
        }