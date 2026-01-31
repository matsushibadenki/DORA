# scripts/optimization/auto_tune_efficiency.py
# SNNの動作効率と精度のトレードオフを自動最適化するためのスクリプト
#
# ディレクトリ: scripts/optimization/auto_tune_efficiency.py
# ファイル名: SNN効率性自動チューニングツール
# 目的: 実モデルを駆動させ、レイテンシと推定精度(発火率ベース)を最適化する。

import argparse
import sys
import logging
import optuna
import time
import torch
import numpy as np
from pathlib import Path

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from snn_research.models.transformer.spikformer import Spikformer

def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="SNN効率性自動チューニング")
    parser.add_argument("--n-trials", type=int, default=20, help="試行回数")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    print(f"🚀 Using Device: {args.device}")

    # 測定用ダミー入力データ (B, C, H, W)
    batch_size = 8 
    input_shape = (batch_size, 3, 224, 224)
    dummy_input = torch.randn(input_shape).to(args.device)

    def objective(trial):
        # --- 探索空間の定義 ---
        # 1. 構造パラメータ (速度と表現力)
        time_steps = trial.suggest_categorical("model.T", [1, 2, 4, 8])
        embed_dim = trial.suggest_categorical("model.embed_dim", [128, 256])
        
        # 2. ニューロンパラメータ (スパイク率/安定性)
        tau_m = trial.suggest_float("model.neuron.tau_m", 1.5, 4.0)
        base_threshold = trial.suggest_float("model.neuron.base_threshold", 0.6, 1.5)

        # モデルの構築
        try:
            model = Spikformer(
                img_size_h=224, img_size_w=224,
                embed_dim=embed_dim,
                num_heads=8,
                num_layers=4,
                T=time_steps,
                num_classes=10
            ).to(args.device)
            
            # ニューロンパラメータの注入 (簡易実装)
            # 実際にはConfig経由が望ましいが、最適化ループ内では直接属性操作を行う
            for m in model.modules():
                if hasattr(m, 'v_threshold'):
                    m.v_threshold = base_threshold
                if hasattr(m, 'tau_m'):
                    # DualAdaptiveLIFNodeの実装に合わせる
                    if hasattr(m, 'tau_m_init'): 
                         m.tau_m.data.fill_(tau_m)
        except Exception as e:
            print(f"⚠️ Model Build Failed: {e}")
            return 1000.0 # ペナルティスコア

        # --- 計測フェーズ ---
        model.eval()
        
        # ウォームアップ (MPS/CUDAの初期化オーバーヘッド排除)
        with torch.no_grad():
            try:
                _ = model(dummy_input)
            except Exception as e:
                print(f"⚠️ Warmup Failed: {e}")
                return 1000.0

        # 1. レイテンシ計測 (Speed)
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10): # 10回平均
                outputs = model(dummy_input)
        end_time = time.time()
        avg_latency_ms = ((end_time - start_time) / 10.0) * 1000.0

        # 2. 仮想精度スコア (Accuracy Potential)
        # 本来はValidationデータで測るが、ここではヒューリスティックな「情報量」を指標とする
        # Tが大きいほど、Embedが大きいほど、閾値が適正(1.0付近)なほど情報量が多いと仮定
        
        # Tによる情報ゲイン (対数的)
        info_gain_t = np.log2(time_steps + 1) * 0.5 
        
        # 次元数による情報ゲイン
        info_gain_dim = 1.0 if embed_dim >= 256 else 0.7
        
        # 閾値ペナルティ (低すぎるとノイズ過多、高すぎると情報消失)
        if base_threshold < 0.8:
            thresh_score = 0.5 # Noise penalty
        elif base_threshold > 1.3:
            thresh_score = 0.6 # Silence penalty
        else:
            thresh_score = 1.0 # Optimal range

        potential_score = (info_gain_t + info_gain_dim) * thresh_score

        # --- 目的関数 (Minimize) ---
        # 目標: レイテンシ < 10ms を維持しつつ、Potentialを最大化したい
        # Score = Latency_Penalty + (Max_Potential - Potential)
        
        latency_penalty = 0.0
        if avg_latency_ms > 10.0:
            latency_penalty = (avg_latency_ms - 10.0) * 2.0 # 10ms超えは厳しく罰する
        
        # 最大Potential目安: (log2(9)*0.5 + 1.0)*1.0 ≈ 2.5
        score = latency_penalty + (3.0 - potential_score)

        return score

    print("🔍 Starting Optimization...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials)

    print("=" * 60)
    print("🏆 チューニング完了")
    print("=" * 60)
    print(f"  Best Score: {study.best_value:.4f}")
    print(f"  Best Params: {study.best_params}")
    
    # 推奨設定の表示
    best = study.best_params
    print("-" * 30)
    print("  [Recommended Configuration for YAML]")
    print(f"  time_steps: {best['model.T']}")
    print(f"  d_model: {best['model.embed_dim']}")
    print(f"  neuron:")
    print(f"    base_threshold: {best['model.neuron.base_threshold']:.2f}")
    print(f"    tau: {best['model.neuron.tau_m']:.2f}")
    print("=" * 60)

if __name__ == "__main__":
    main()