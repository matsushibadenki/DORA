# ファイルパス: snn_research/learning_rules/stdp.py
# 日本語タイトル: STDP Rule (Fixed with Auto-Shape Inference)
# 目的・内容:
#   in_features/out_features の自動推論に対応し、既存コードからの呼び出しエラーを解消。
#   torch.exp の引数型エラー修正。

from __future__ import annotations

import torch
from torch import Tensor
from typing import Tuple, Dict, Any, Optional

from snn_research.learning_rules.base_rule import PlasticityRule


class STDP(PlasticityRule):
    """
    Trace-based STDP Implementation.
    """
    
    # バッファの型ヒント
    trace_pre: Tensor
    trace_post: Tensor

    def __init__(
        self,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        learning_rate: float = 1e-4,
        tau_pre: float = 20.0,
        tau_post: float = 20.0,
        w_max: float = 1.0,
        w_min: float = -1.0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.lr = learning_rate
        
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.w_max = w_max
        self.w_min = w_min

        # 次元が指定されている場合は初期化、なければNoneでupdate時に初期化
        if in_features is not None and out_features is not None:
            self.register_buffer('trace_pre', torch.zeros(1, in_features))
            self.register_buffer('trace_post', torch.zeros(1, out_features))
        else:
            # 仮のバッファ登録（state_dict互換性のため空テンソル等を登録しておくと安全だが、
            # ここでは初期化フラグとしてサイズ0のテンソルを使用）
            self.register_buffer('trace_pre', torch.empty(0))
            self.register_buffer('trace_post', torch.empty(0))

    def _init_traces_if_needed(self, pre_shape: Tuple[int, ...], post_shape: Tuple[int, ...], device: torch.device) -> None:
        """必要な場合にトレースバッファを初期化・再確保する"""
        batch_size = pre_shape[0]
        in_feat = pre_shape[1]
        out_feat = post_shape[1]

        # 未初期化(numel==0) または バッチサイズ不一致 または 特徴量次元の不一致
        needs_init = (self.trace_pre.numel() == 0) or \
                     (self.trace_pre.shape[0] != batch_size) or \
                     (self.trace_pre.shape[1] != in_feat)

        if needs_init:
            self.in_features = in_feat
            self.out_features = out_feat
            self.trace_pre = torch.zeros(batch_size, in_feat, device=device)
            self.trace_post = torch.zeros(batch_size, out_feat, device=device)

    def update(
        self,
        pre_spikes: Tensor,
        post_spikes: Tensor,
        current_weights: Tensor,
        dt: float = 1.0,
        **kwargs: Any
    ) -> Tuple[Optional[Tensor], Dict[str, Any]]:
        """
        STDP更新則の適用。
        """
        # トレースの初期化チェック
        self._init_traces_if_needed(pre_spikes.shape, post_spikes.shape, pre_spikes.device)

        batch_size = pre_spikes.shape[0]

        # Decay traces
        # torch.expにはTensorが必要。dt/tauはfloatなのでTensor化する。
        decay_pre = torch.exp(torch.tensor(-dt / self.tau_pre, device=pre_spikes.device))
        decay_post = torch.exp(torch.tensor(-dt / self.tau_post, device=pre_spikes.device))

        # x[t] = x[t-1] * decay + spike[t]
        self.trace_pre = self.trace_pre * decay_pre + pre_spikes
        self.trace_post = self.trace_post * decay_post + post_spikes

        # 重み更新量の計算 (Batch平均)
        # delta_ltp[o, i] = sum_b( post_spikes[b, o] * trace_pre[b, i] )
        delta_ltp = torch.einsum('bo,bi->oi', post_spikes, self.trace_pre)
        
        # delta_ltd[o, i] = sum_b( trace_post[b, o] * pre_spikes[b, i] )
        delta_ltd = torch.einsum('bo,bi->oi', self.trace_post, pre_spikes)

        delta_w = self.lr * (delta_ltp - delta_ltd) / batch_size

        # ログデータの作成
        logs = {
            "mean_ltp": delta_ltp.mean().item(),
            "mean_ltd": delta_ltd.mean().item(),
            "mean_delta": delta_w.mean().item()
        }

        return delta_w, logs

    def reset_state(self) -> None:
        if self.trace_pre.numel() > 0:
            self.trace_pre.fill_(0.0)
            self.trace_post.fill_(0.0)


# --- Backward Compatibility ---
STDPRule = STDP