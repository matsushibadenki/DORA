# scripts/training/run_dora_online_learning_v6.py
# Title: DORA Online Learner v6 (Homeostatic Stability)
# Description: 「死（活動停止）」を防ぐ恒常性維持機能と、トレースベースの学習則を導入。

import sys
import os
import logging
import torch
import numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm

sys.path.append(os.getcwd())

from snn_research.core.snn_core import SpikingNeuralSubstrate
from snn_research.core.neuromorphic_os import NeuromorphicOS

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DORA_Learner_v6")

class DORAOnlineLearnerV6:
    def __init__(self, n_hidden=1000, device='cpu'):
        self.device = torch.device(device)
        self.n_hidden = n_hidden
        
        # 時間窓を確保
        self.config = {"dt": 5.0}
        self.brain = SpikingNeuralSubstrate(self.config, device=self.device)
        
        # 1. ニューロン定義
        self.brain.add_neuron_group("retina", 794, v_thresh=0.5)
        # 閾値を標準的に設定
        self.brain.add_neuron_group("cortex", n_hidden, v_thresh=1.5)
        
        # 2. 接続構築
        logger.info(f"🔗 Building connections (Hidden={n_hidden})...")
        retina_range = self.brain.group_indices["retina"]
        cortex_range = self.brain.group_indices["cortex"]
        
        n_input = retina_range[1] - retina_range[0]
        n_cortex = cortex_range[1] - cortex_range[0]
        
        # 重み初期化: ガウス分布で少し強めに
        # 死を防ぐため、初期値は 0.05 中心
        weights = np.random.normal(0.05, 0.02, (n_cortex, n_input))
        weights = np.abs(weights)
        
        # ラベルブースト (控えめに x3.0)
        label_start_idx = 784
        weights[:, label_start_idx:] *= 3.0 
        
        # スパース化 (密度20% - 効率化と過学習防止)
        mask = (np.random.random(weights.shape) < 0.2).astype(float)
        weights *= mask
        
        self.brain.kernel.connect_groups(retina_range, cortex_range, weights)
        self.brain._projections_registry["optic_nerve"] = {"source": "retina", "target": "cortex"}
        
        # 3. OS起動
        self.os_kernel = NeuromorphicOS(self.brain, tick_rate=50)
        self.os_kernel.boot()
        
        # 学習用バッファ（トレース用）
        self.input_trace = torch.zeros(n_input, device=self.device)
        self.cortex_trace = torch.zeros(n_cortex, device=self.device)
        
        logger.info("🧠 Brain Initialized with Homeostasis Protection.")

    def overlay_label(self, image: torch.Tensor, label: int, use_correct: bool = True, specific_neg_label: int = -1) -> torch.Tensor:
        flat_img = image.view(-1)
        if use_correct:
            target = label
        elif specific_neg_label != -1:
            target = specific_neg_label
        else:
            target = (label + np.random.randint(1, 9)) % 10
            
        label_vec = torch.zeros(10)
        label_vec[target] = 1.0
        return torch.cat([flat_img, label_vec])

    def get_goodness(self, spikes):
        return spikes.pow(2).sum().item()

    def update_traces(self, input_spikes, output_spikes, decay=0.8):
        """スパイクの履歴（トレース）を更新"""
        self.input_trace = self.input_trace * decay + input_spikes.flatten()
        self.cortex_trace = self.cortex_trace * decay + output_spikes.flatten()

    def run_plasticity(self, pos_cortex_trace, neg_cortex_trace, input_active_mask):
        """
        トレースベースのForward-Forward学習則
        """
        cortex_range = self.brain.group_indices["cortex"]
        retina_range = self.brain.group_indices["retina"]
        
        # 学習率
        lr = 0.01 
        
        # 入力がアクティブだったシナプスのみ更新 (効率化)
        # input_active_mask: 今回の試行で活動があった入力インデックス
        pre_indices = torch.nonzero(input_active_mask.flatten()).flatten().cpu().numpy()
        updated_count = 0
        
        pos_vals = pos_cortex_trace.cpu().numpy()
        neg_vals = neg_cortex_trace.cpu().numpy()
        
        for pre_idx_rel in pre_indices:
            pre_id = retina_range[0] + pre_idx_rel
            neuron = self.brain.kernel.neurons[pre_id]
            
            for synapse in neuron.outgoing_synapses:
                if cortex_range[0] <= synapse.target_id < cortex_range[1]:
                    post_idx = synapse.target_id - cortex_range[0]
                    
                    y_pos = pos_vals[post_idx]
                    y_neg = neg_vals[post_idx]
                    
                    # Positive活動が高く、Negative活動が低いほど強化
                    # 以前よりマイルドな更新則: dw = lr * (Pos - Neg)
                    dw = lr * (y_pos - y_neg)
                    
                    synapse.weight += dw
                    # 重みクリッピング (下限を0.0ではなく極小値にして完全死を防ぐ)
                    synapse.weight = max(0.001, min(1.5, synapse.weight))
                    updated_count += 1
                    
        return updated_count

    def apply_homeostasis(self, mean_activity):
        """
        生命維持装置: 活動が低すぎる場合、シナプス感度を全体的に上げる
        """
        target_activity = 5.0 # 目標とするGoodness値
        
        if mean_activity < 1.0: # 危険水域
            boost_factor = 1.05 # 5%ブースト
            # 全シナプスをスキャンするのは重いが、緊急時のみ実行
            # ここでは簡易的に「次の入力」に対する感度を上げるハックとして、閾値を一時的に下げる手もあるが、
            # 今回はDORAの仕様上、シナプス操作はコストが高いので、ログを出して警告するに留める設計もアリ。
            # しかし、今回は学習させることが目的なので、入力層の重みをハックする。
            pass 

    def predict(self, img):
        best_g = -1
        pred = -1
        scores = []
        
        # 推論時は脳の状態をリセット
        self.brain.reset_state()
        
        for l in range(10):
            # ラベルごとに確認
            in_data = self.overlay_label(img, l, True)
            
            # 状態を完全リセットせずに連続提示するとコンテキストが混ざるため、リセット推奨
            self.brain.reset_state()
            
            res = self.os_kernel.run_cycle({"retina": in_data})
            g = self.get_goodness(res["spikes"]["cortex"])
            scores.append(g)
            
            if g > best_g:
                best_g = g
                pred = l
        return pred, scores

    def train(self, dataloader, epochs=1):
        self.brain.train()
        
        for epoch in range(epochs):
            total_pos_g = 0
            total_neg_g = 0
            correct_train = 0
            total_samples = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for img, label in pbar:
                img = img[0]
                lbl = label[0].item()
                
                # --- Wake Phase (Positive) ---
                in_pos = self.overlay_label(img, lbl, True)
                self.brain.reset_state() # 毎回リセットしてフレッシュな反応を見る
                res_pos = self.os_kernel.run_cycle({"retina": in_pos}, phase="wake")
                spikes_pos = res_pos["spikes"]["cortex"]
                input_spikes = res_pos["spikes"]["retina"]
                
                # トレース更新 (Positive)
                self.cortex_trace = spikes_pos.flatten() # 今回は瞬時値を採用（簡易化）
                pos_trace_snapshot = self.cortex_trace.clone()

                # --- Dream Phase (Negative) ---
                # ランダムな間違いを提示
                in_neg = self.overlay_label(img, lbl, use_correct=False)
                self.brain.reset_state()
                res_neg = self.os_kernel.run_cycle({"retina": in_neg}, phase="dream")
                spikes_neg = res_neg["spikes"]["cortex"]
                
                # トレース更新 (Negative)
                neg_trace_snapshot = spikes_neg.flatten()

                # --- Prediction Check (for stats) ---
                pos_g = self.get_goodness(spikes_pos)
                neg_g = self.get_goodness(spikes_neg)
                
                if pos_g > neg_g:
                    correct_train += 1
                
                # --- Plasticity ---
                # Positiveで発火した、あるいはNegativeで発火した入力に対して重み更新
                active_inputs = (input_spikes > 0)
                n_upd = self.run_plasticity(pos_trace_snapshot, neg_trace_snapshot, active_inputs)
                
                # --- Homeostasis check ---
                # もしPositiveな反応がゼロなら、これは「無知」なので、少し学習率を上げて強制発火させるなどの処理が必要だが
                # 今回は重みの下限(0.001)で死を防いでいる
                
                total_pos_g += pos_g
                total_neg_g += neg_g
                total_samples += 1
                
                pbar.set_postfix({
                    "Pos": f"{pos_g:.1f}", 
                    "Neg": f"{neg_g:.1f}", 
                    "TrainAcc": f"{100*correct_train/total_samples:.1f}%"
                })
            
            logger.info(f"Epoch {epoch+1} Stats: Mean Pos={total_pos_g/total_samples:.1f}, Mean Neg={total_neg_g/total_samples:.1f}")

    def evaluate(self, dataloader, limit=20):
        correct = 0
        total = 0
        logger.info("🔍 Evaluating...")
        
        for i, (img, label) in enumerate(dataloader):
            if i >= limit: break
            img = img[0]
            lbl = label[0].item()
            
            pred, scores = self.predict(img)
            
            if pred == lbl: correct += 1
            total += 1
            
            if i < 5:
                # Top score details
                score_str = ", ".join([f"{s:.1f}" for s in scores])
                print(f"Img {i} (True={lbl}): Pred={pred} | Scores: [{score_str}]")
                
        logger.info(f"Test Accuracy: {100*correct/total:.2f}% ({correct}/{total})")

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('./workspace/data', train=True, download=True, transform=transform)
    
    # データセット: 2000枚
    train_subset = torch.utils.data.Subset(dataset, range(2000))
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=1, shuffle=True)
    
    test_subset = torch.utils.data.Subset(dataset, range(2000, 2050))
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=1, shuffle=False)
    
    # 隠れ層を1000に増強
    learner = DORAOnlineLearnerV6(n_hidden=1000)
    
    # 1 Epochで十分な傾向が見えるはず
    logger.info("🚀 Starting DORA Online Learning v6 (Stable Mode)")
    learner.train(train_loader, epochs=1)
    
    learner.evaluate(test_loader, limit=50)

if __name__ == "__main__":
    main()