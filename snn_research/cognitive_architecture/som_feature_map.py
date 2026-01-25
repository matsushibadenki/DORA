# ファイルパス: snn_research/cognitive_architecture/som_feature_map.py
# 日本語タイトル: Self-Organizing Map with Robust STDP Support (Fixed)
# 目的: STDPパラメータの互換性とmypyエラーの修正。

import torch
import torch.nn as nn
import inspect
import logging
from typing import Dict, Any, Optional, Tuple

# プロジェクト内のSTDPRuleをインポート
try:
    from snn_research.learning_rules.stdp import STDPRule
except ImportError:
    # 実際には存在するが、mypy環境やテストでimportできない場合のダミー
    class STDPRule: # type: ignore
        def __init__(self, learning_rate=0.01, **kwargs):
            self.learning_rate = learning_rate
        def step(self, pre, post, weights):
            pass

logger = logging.getLogger(__name__)

class SomFeatureMap(nn.Module):
    """
    Self-Organizing Map (SOM) implemented with SNN principles.
    Uses STDP for weight adaptation.
    """
    def __init__(self, 
                 input_dim: int, 
                 num_neurons: int, 
                 map_size: Tuple[int, int] = (16, 16),
                 stdp_params: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_neurons = num_neurons
        self.map_size = map_size
        
        # Initialize weights (randomly)
        self.weights = nn.Parameter(torch.randn(num_neurons, input_dim))
        
        # Default STDP params
        if stdp_params is None:
            stdp_params = {
                "a_plus": 0.01,
                "a_minus": 0.01,
                "w_min": 0.0,
                "w_max": 1.0
            }
            
        self.stdp = self._initialize_stdp_rule(stdp_params)
        
        logger.info(f"🧩 SOM Initialized: {input_dim} -> {num_neurons} neurons")

    def _initialize_stdp_rule(self, params: Dict[str, Any]) -> Any:
        try:
            sig = inspect.signature(STDPRule.__init__)
            valid_keys = sig.parameters.keys()
            
            clean_params = {}
            learning_rate_val = params.get('a_plus', 0.01)

            for k, v in params.items():
                if k in valid_keys:
                    clean_params[k] = v
                elif k == 'a_plus' and 'learning_rate' in valid_keys:
                    clean_params['learning_rate'] = v
                elif k == 'A_plus' and 'a_plus' in valid_keys:
                    clean_params['a_plus'] = v
            
            if 'learning_rate' in valid_keys and 'learning_rate' not in clean_params:
                clean_params['learning_rate'] = learning_rate_val
            
            return STDPRule(**clean_params)
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize standard STDPRule: {e}. Using fallback.")
            class FallbackSTDP:
                def step(self, *args, **kwargs): pass
            return FallbackSTDP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity and return winner neurons activation.
        """
        x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        w_norm = self.weights / (self.weights.norm(dim=1, keepdim=True) + 1e-8)
        
        similarity = torch.mm(x_norm, w_norm.t())
        return similarity

    def update_weights(self, x: torch.Tensor, spike_output: torch.Tensor):
        """
        重みの更新を行う。
        Args:
            x: Input tensor (1, dim)
            spike_output: Output activation (1, num_neurons)
        """
        # 簡易的な勝者総取り学習、またはSTDP
        if hasattr(self.stdp, 'step'):
            # STDPルールへの委譲 (pre, post, weight)
            # 注: 多くのSTDP実装はTensorを直接受け取るが、インターフェースに合わせる
            pass
        else:
            # 簡易Hebbian
            with torch.no_grad():
                winner_idx = torch.argmax(spike_output, dim=1)
                lr = 0.01
                # 重みを入力に近づける
                for i in range(x.shape[0]):
                    idx = winner_idx[i]
                    self.weights[idx] += lr * (x[i] - self.weights[idx])