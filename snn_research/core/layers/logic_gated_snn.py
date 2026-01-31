# snn_research/core/layers/logic_gated_snn.py
# Title: Phase-Critical SCAL (Unified v3.4)
# Description:
#   Statistical Centroid Alignment Learning (SCAL) の決定版実装。
#   閾値の超安定化制御、重心モーメンタム更新、マルチスケール発火制御を統合。

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, cast, Optional

class PhaseCriticalSCAL(nn.Module):
    """
    Phase-Critical SCAL (Statistical Centroid Alignment Learning) v3.4
    
    特徴:
    - Bipolar変換と直交初期化による特徴分離
    - クラスごとの重心(Centroid)へのアライメント学習
    - 分散とホメオスタシスに基づくハイブリッド閾値制御
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        mode: str = 'readout',
        time_steps: int = 10,
        gain: float = 50.0,
        beta_membrane: float = 0.9,
        v_th_init: float = 25.0,
        v_th_min: float = 5.0,
        v_th_max: float = 100.0,
        gamma_th: float = 0.01,
        target_spike_rate: float = 0.15,
        spike_rate_control_strength: float = 0.001,
        learning_rate: float = 0.01,
        momentum: float = 0.95
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.time_steps = time_steps
        self.gain = gain
        self.beta_membrane = beta_membrane
        self.learning_rate = learning_rate
        self.momentum_coef = momentum
        
        self.v_th_min = v_th_min
        self.v_th_max = v_th_max
        self.gamma_th = gamma_th
        self.target_spike_rate = target_spike_rate
        self.spike_rate_control_strength = spike_rate_control_strength
        
        # 投影層 (特徴抽出)
        self.projection = nn.Linear(in_features, in_features, bias=False)
        nn.init.orthogonal_(self.projection.weight, gain=1.0)
        
        # SCAL重心 (Synaptic Weights equivalent)
        self.register_buffer('centroids', torch.randn(out_features, in_features))
        self.centroids: torch.Tensor
        self.centroids.div_(math.sqrt(in_features))
        
        # 適応パラメータ
        self.register_buffer('adaptive_threshold', torch.full((out_features,), v_th_init))
        self.adaptive_threshold: torch.Tensor 
        
        self.register_buffer('spike_rate_ema', torch.full((out_features,), target_spike_rate))
        self.spike_rate_ema: torch.Tensor 
        
        # 状態モニタリング
        self.stats = {
            'spike_rate': target_spike_rate,
            'mean_threshold': v_th_init,
            'threshold_min': v_th_min,
            'threshold_max': v_th_max,
        }

    def reset_state(self):
        pass

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = x.size(0)
        device = x.device
        
        # 1. Bipolar変換 & 特徴抽出
        x_bipolar = 2.0 * x - 1.0
        features = self.projection(x_bipolar)
        features = torch.tanh(features)
        
        # 2. 類似度計算 (Centroid Alignment)
        centroids_t = cast(torch.Tensor, self.centroids)
        features_norm = F.normalize(features, p=2, dim=1, eps=1e-8)
        centroids_norm = F.normalize(centroids_t, p=2, dim=1, eps=1e-8)
        
        # (Batch, In) @ (Out, In)^T -> (Batch, Out)
        cosine_sim = torch.mm(features_norm, centroids_norm.t())
        
        # 3. 高ゲイン入力
        s_prime = self.gain * cosine_sim
        
        # 4. 時系列積分 (LIF Dynamics)
        spike_trains = []
        V_mem_history = []
        V_current = torch.zeros(batch_size, self.out_features, device=device)
        
        th_tensor = cast(torch.Tensor, self.adaptive_threshold)
        
        for _ in range(self.time_steps):
            V_current = self.beta_membrane * V_current + s_prime
            
            # 発火判定
            threshold = th_tensor.unsqueeze(0)
            
            if self.training:
                # Surrogate Gradient用のSigmoid
                surrogate_grad = torch.sigmoid((V_current - threshold))
                spikes = (surrogate_grad > torch.rand_like(surrogate_grad)).float()
            else:
                spikes = (V_current > threshold).float()
            
            # Reset
            V_current = V_current * (1.0 - spikes)
            
            spike_trains.append(spikes)
            V_mem_history.append(V_current.clone())
        
        spike_trains_stack = torch.stack(spike_trains, dim=2)
        V_mem_stack = torch.stack(V_mem_history, dim=2)
        output_rate = spike_trains_stack.mean(dim=2)
        
        # 統計更新 (EMA)
        with torch.no_grad():
            current_mean_rate = output_rate.mean(dim=0)
            self.spike_rate_ema.mul_(0.95).add_(current_mean_rate * 0.05)
            
            self.stats['spike_rate'] = output_rate.mean().item()
            self.stats['mean_threshold'] = th_tensor.mean().item()
            
        return {
            'output': output_rate,
            'spikes': spike_trains_stack,
            'membrane_potential': V_mem_stack,
            'features': features
        }

    def update_plasticity(
        self, 
        pre_activity: torch.Tensor, 
        post_output: Dict[str, torch.Tensor], 
        target: torch.Tensor, 
        learning_rate: Optional[float] = None
    ):
        """Phase-Critical Plasticity Update"""
        lr = learning_rate if learning_rate is not None else self.learning_rate
        
        with torch.no_grad():
            # 特徴量再計算 (推論時と同じ変換)
            x_bipolar = 2.0 * pre_activity - 1.0
            features = torch.tanh(self.projection(x_bipolar))
            
            unique_classes = torch.unique(target)
            
            centroids_t = cast(torch.Tensor, self.centroids)
            th_tensor = cast(torch.Tensor, self.adaptive_threshold)
            ema_tensor = cast(torch.Tensor, self.spike_rate_ema)
            
            for c in unique_classes:
                c_idx = int(c.item())
                mask = (target == c)
                class_features = features[mask]
                
                # --- 1. 重心更新 (SCAL Logic) ---
                current_centroid = centroids_t[c_idx]
                new_centroid = class_features.mean(dim=0)
                # モーメンタム的更新: 現在の重心位置から新しい平均位置へ少し移動
                centroids_t[c_idx] = current_centroid + (lr * 0.5) * (new_centroid - current_centroid)
                
                # --- 2. 閾値制御 (Hybrid: Variance + Homeostasis) ---
                if class_features.size(0) > 1:
                    variance = torch.var(class_features, dim=0)
                    variance_norm = torch.norm(variance)
                    var_factor = 1.0 - self.gamma_th * variance_norm
                else:
                    var_factor = torch.tensor(1.0, device=features.device)

                # ホメオスタシス制御: 目標発火率との誤差で補正
                rate_error = ema_tensor[c_idx] - self.target_spike_rate
                homeo_factor = 1.0 + self.spike_rate_control_strength * rate_error
                
                # 統合係数 & クランプ
                total_factor = var_factor * homeo_factor
                total_factor = torch.clamp(total_factor, 0.995, 1.005) # 変動を0.5%以内に制限（安定化の鍵）
                
                th_tensor[c_idx] *= total_factor
                th_tensor[c_idx] = torch.clamp(
                    th_tensor[c_idx], self.v_th_min, self.v_th_max
                )
            
            # 正規化 (Centroidsは常に単位ベクトル付近に保つ)
            centroids_t.div_(centroids_t.norm(dim=1, keepdim=True) + 1e-8)

    def get_phase_critical_metrics(self) -> Dict[str, float]:
        th_tensor = cast(torch.Tensor, self.adaptive_threshold)
        return {
            'spike_rate': self.stats['spike_rate'],
            'mean_threshold': self.stats['mean_threshold'],
            'threshold_min': th_tensor.min().item(),
            'threshold_max': th_tensor.max().item(),
        }

# 互換性エイリアス
LogicGatedSNN = PhaseCriticalSCAL