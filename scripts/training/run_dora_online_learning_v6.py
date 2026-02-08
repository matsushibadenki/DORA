# scripts/training/run_dora_online_learning_v6.py
# Japanese Title: DORA オンライン学習 v10.0 (強結合・高反応版)
# Description: スパイクの「数」ではなく結合の「強さ」で発火を保証するよう修正。入力特徴に対する感度を大幅に高めたバージョン。

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
logger = logging.getLogger("DORA_Learner_v10_HighGain")

class DORAOnlineLearnerV10:
    def __init__(self, n_hidden=1000, device='cpu'):
        self.device = torch.device(device)
        self.n_hidden = n_hidden
        
        # タイムステップ設定
        # 標準的なSNNパラメータに戻す
        self.config = {
            "dt": 1.0, 
            "t_ref": 5.0,
            "tau_m": 20.0,
        }
        self.brain = SpikingNeuralSubstrate(self.config, device=self.device)
        
        # 1. ニューロン定義
        self.brain.add_neuron_group("retina", 794, v_thresh=0.5)
        
        # 閾値 1.0
        self.brain.add_neuron_group("cortex", n_hidden, v_thresh=1.0)
        
        # 2. 接続構築
        logger.info(f"🔗 Building connections (Hidden={n_hidden})...")
        retina_range = self.brain.group_indices["retina"]
        cortex_range = self.brain.group_indices["cortex"]
        
        n_input = retina_range[1] - retina_range[0]
        n_cortex = cortex_range[1] - cortex_range[0]
        
        # 【修正1】重みの強化
        # MNISTの有効画素数が150、密度10%なら、1ニューロンあたりの入力は約15個。
        # 15個 × 0.08 = 1.2 > 閾値1.0
        # これにより、画像入力だけで確実に発火する
        weights = np.random.uniform(0.05, 0.10, (n_cortex, n_input))
        
        # ラベル部分: さらに強く (1発で発火に寄与)
        label_start_idx = 784
        weights[:, label_start_idx:] = 2.0 
        
        # 接続密度 (10%)
        mask = (np.random.random(weights.shape) < 0.10).astype(float)
        weights *= mask
        # ラベルは全結合
        weights[:, label_start_idx:] = 2.0
        
        self.brain.kernel.connect_groups(retina_range, cortex_range, weights)
        
        # 【修正2】側抑制 (-1.0)
        # 発火が強まるので、抑制も確実に効かせる
        inhibition_weights = -1.0 * np.ones((n_cortex, n_cortex))
        np.fill_diagonal(inhibition_weights, 0)
        
        # 抑制密度 (30%)
        inhib_mask = (np.random.random(inhibition_weights.shape) < 0.30).astype(float)
        inhibition_weights *= inhib_mask
        
        self.brain.kernel.connect_groups(cortex_range, cortex_range, inhibition_weights)
        
        self.brain._projections_registry["optic_nerve"] = {"source": "retina", "target": "cortex"}
        self.brain._projections_registry["lateral_inhibition"] = {"source": "cortex", "target": "cortex"}
        
        # 3. OS起動
        self.os_kernel = NeuromorphicOS(self.brain, tick_rate=50)
        self.os_kernel.boot()
        
        logger.info("🧠 Brain Initialized: High Gain & Sparse Mode (W~0.08).")

    def overlay_label(self, image: torch.Tensor, label: int, use_correct: bool = True) -> torch.Tensor:
        """入力の二値化のみ行う（値のブーストはしない）"""
        flat_img = (image.view(-1) > 0.3).float()
        
        if not use_correct:
            label_candidates = list(range(10))
            if label in label_candidates:
                label_candidates.remove(label)
            label = np.random.choice(label_candidates)
            
        label_vec = torch.zeros(10)
        label_vec[label] = 1.0 
        return torch.cat([flat_img, label_vec])

    def get_goodness(self, spikes):
        return spikes.sum().item()

    def _safe_numpy(self, tensor_spikes):
        vals = tensor_spikes.detach().cpu().numpy().flatten()
        if vals.size != self.n_hidden:
            safe_vals = np.zeros(self.n_hidden, dtype=np.float32)
            if vals.size > 0:
                limit = min(vals.size, self.n_hidden)
                safe_vals[:limit] = vals[:limit]
            return safe_vals
        return vals

    def run_plasticity(self, pos_spikes, neg_spikes, input_spikes):
        """学習則"""
        cortex_range = self.brain.group_indices["cortex"]
        retina_range = self.brain.group_indices["retina"]
        
        lr = 0.05 
        weight_decay = 0.001 
        
        pre_indices = torch.nonzero(input_spikes.flatten() > 0).flatten().cpu().numpy()
        pos_vals = self._safe_numpy(pos_spikes)
        neg_vals = self._safe_numpy(neg_spikes)
        
        updated_count = 0
        active_post_indices = np.where((pos_vals > 0) | (neg_vals > 0))[0]
        
        if len(active_post_indices) == 0:
            return 0

        for pre_idx_rel in pre_indices:
            pre_id = retina_range[0] + pre_idx_rel
            if pre_id >= len(self.brain.kernel.neurons): continue
            
            neuron = self.brain.kernel.neurons[pre_id]
            
            for synapse in neuron.outgoing_synapses:
                if cortex_range[0] <= synapse.target_id < cortex_range[1]:
                    post_idx = synapse.target_id - cortex_range[0]
                    
                    if post_idx in active_post_indices:
                        val_p = pos_vals[post_idx]
                        val_n = neg_vals[post_idx]
                        
                        dw = lr * (val_p - val_n)
                        
                        synapse.weight += dw
                        synapse.weight -= weight_decay * synapse.weight
                        
                        # 重み制限: 2.0まで許容
                        synapse.weight = max(0.001, min(2.0, synapse.weight))
                        updated_count += 1
                        
        return updated_count

    def predict(self, img):
        best_g = -1
        pred = -1
        scores = []
        
        for l in range(10):
            in_data = self.overlay_label(img, l, True)
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
            correct_train = 0
            total_samples = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for img, label in pbar:
                img = img[0]
                lbl = label[0].item()
                
                # --- Positive Phase ---
                in_pos = self.overlay_label(img, lbl, True)
                self.brain.reset_state()
                res_pos = self.os_kernel.run_cycle({"retina": in_pos}, phase="wake")
                spikes_pos = res_pos["spikes"]["cortex"]
                input_spikes = res_pos["spikes"]["retina"]
                
                # --- Negative Phase ---
                in_neg = self.overlay_label(img, lbl, False)
                self.brain.reset_state()
                res_neg = self.os_kernel.run_cycle({"retina": in_neg}, phase="dream")
                spikes_neg = res_neg["spikes"]["cortex"]

                # --- Learning ---
                self.run_plasticity(spikes_pos, spikes_neg, input_spikes)
                
                pos_g = self.get_goodness(spikes_pos)
                neg_g = self.get_goodness(spikes_neg)
                
                if pos_g > neg_g and pos_g > 0:
                    correct_train += 1
                
                total_samples += 1
                
                if total_samples % 10 == 0:
                    pbar.set_postfix({
                        "Pos": f"{pos_g:.0f}", 
                        "Neg": f"{neg_g:.0f}", 
                        "Acc": f"{100*correct_train/total_samples:.1f}%"
                    })

    def evaluate(self, dataloader, limit=50):
        correct = 0
        total = 0
        logger.info(f"🔍 Evaluating top {limit} samples...")
        
        for i, (img, label) in enumerate(dataloader):
            if i >= limit: break
            img = img[0]
            lbl = label[0].item()
            
            pred, scores = self.predict(img)
            
            if max(scores) == 0:
                pred = -1
            
            if pred == lbl: correct += 1
            total += 1
            
            if i < 5:
                print(f"Sample {i}: True={lbl}, Pred={pred}, Scores={np.round(scores, 1)}")
                
        accuracy = 100 * correct / total
        logger.info(f"✅ Final Test Accuracy: {accuracy:.2f}%")

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.MNIST('./workspace/data', train=True, download=True, transform=transform)
    
    # 2000枚
    train_subset = torch.utils.data.Subset(dataset, range(2000))
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=1, shuffle=True)
    
    test_subset = torch.utils.data.Subset(dataset, range(2000, 2050))
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=1, shuffle=False)
    
    learner = DORAOnlineLearnerV10(n_hidden=1000)
    
    logger.info("🚀 Starting DORA Online Learning (v10.0 High Gain)")
    learner.train(train_loader, epochs=1)
    learner.evaluate(test_loader, limit=50)

if __name__ == "__main__":
    main()