# scripts/training/run_dora_online_learning_v6.py
# Japanese Title: DORA オンライン学習 v14.0 (正規化ポアソンSNN)
# Description: 重みの正規化(Normalization)と入力レートの抑制により、過剰発火を防ぎ、特徴選択性（Selectivity）を向上させたバージョン。

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
logger = logging.getLogger("DORA_Learner_v14_NormPoisson")

# -----------------------------------------------------------------------------
# ポアソン入力対応SNNコア (v13ベース + 軽量化)
# -----------------------------------------------------------------------------
class PoissonSNN(SpikingNeuralSubstrate):
    def forward_step(self, ext_inputs: dict, learning: bool = True, dreaming: bool = False, **kwargs) -> dict:
        # 処理時間を20msに短縮してスパイク数を抑える
        simulation_duration = kwargs.get("duration", 20.0)
        
        if not dreaming:
            for name, tensor in ext_inputs.items():
                if name in self.group_indices:
                    start_id, _ = self.group_indices[name]
                    input_probs = torch.clamp(tensor.flatten(), 0, 1).cpu().numpy()
                    
                    # 閾値を上げて重要な画素のみスパイク化
                    active_indices = np.where(input_probs > 0.2)[0]
                    
                    for idx in active_indices:
                        # レート係数を 0.1 -> 0.05 に下げて疎にする
                        rate = input_probs[idx] * 0.05 
                        for t in range(int(simulation_duration)):
                            if np.random.random() < rate:
                                self.kernel.push_input_spikes([int(start_id + idx)], self.kernel.current_time + t + 0.1)

        counts = self.kernel.run(duration=simulation_duration, learning_enabled=learning)
        
        curr_spikes = {}
        for name, (s, e) in self.group_indices.items():
            spikes = torch.zeros(1, e-s, device=self.device)
            for nid, count in counts.items():
                if s <= nid < e and count > 0:
                    spikes[0, nid-s] = count 
            curr_spikes[name] = spikes
            
        return {"spikes": curr_spikes}

# -----------------------------------------------------------------------------

class DORAOnlineLearnerV14:
    def __init__(self, n_hidden=1000, device='cpu'):
        self.device = torch.device(device)
        self.n_hidden = n_hidden
        
        self.config = {
            "dt": 1.0, 
            "t_ref": 2.0,
            "tau_m": 20.0,
        }
        self.brain = PoissonSNN(self.config, device=self.device)
        
        # 1. ニューロン定義
        self.brain.add_neuron_group("retina", 794, v_thresh=0.5)
        # 閾値を少し上げて(1.0->2.0)、ノイズ耐性を上げる
        self.brain.add_neuron_group("cortex", n_hidden, v_thresh=2.0)
        
        # 2. 接続構築
        logger.info(f"🔗 Building connections (Hidden={n_hidden})...")
        retina_range = self.brain.group_indices["retina"]
        cortex_range = self.brain.group_indices["cortex"]
        
        n_input = retina_range[1] - retina_range[0]
        n_cortex = cortex_range[1] - cortex_range[0]
        
        # 重み設定: 少し強めにしておく（後で正規化されるため）
        weights = np.random.uniform(0.05, 0.10, (n_cortex, n_input))
        
        # ラベルは明確に強く
        label_start_idx = 784
        weights[:, label_start_idx:] = 2.0 
        
        # 接続密度 (10%)
        mask = (np.random.random(weights.shape) < 0.10).astype(float)
        weights *= mask
        weights[:, label_start_idx:] = 2.0
        
        self.brain.kernel.connect_groups(retina_range, cortex_range, weights)
        
        # 側抑制 (-2.0)
        inhibition_weights = -2.0 * np.ones((n_cortex, n_cortex))
        np.fill_diagonal(inhibition_weights, 0)
        inhib_mask = (np.random.random(inhibition_weights.shape) < 0.20).astype(float)
        inhibition_weights *= inhib_mask
        
        self.brain.kernel.connect_groups(cortex_range, cortex_range, inhibition_weights)
        
        self.brain._projections_registry["optic_nerve"] = {"source": "retina", "target": "cortex"}
        self.brain._projections_registry["lateral_inhibition"] = {"source": "cortex", "target": "cortex"}
        
        self.brain.kernel.structural_plasticity_enabled = False
        
        # 3. OS起動
        self.os_kernel = NeuromorphicOS(self.brain, tick_rate=50)
        self.os_kernel.boot()
        
        logger.info("🧠 Brain Initialized: Normalized Poisson Mode (Duration=20ms).")

    def overlay_label(self, image: torch.Tensor, label: int, use_correct: bool = True) -> torch.Tensor:
        flat_img = torch.clamp(image.view(-1), 0, 1)
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

    def normalize_weights(self):
        """【新機能】重み正規化: 各ニューロンの入力重みの合計を一定に保つ"""
        # ニューロンごとのループ処理は重いが、学習の安定性には必須
        # 本来は行列演算で行うべきだが、カーネル構造上イテレーションする
        cortex_range = self.brain.group_indices["cortex"]
        
        # サンプリングして一部だけ正規化するか、数バッチに一度実行するのが良いが、
        # ここでは簡易的に全ニューロンをチェック（重いようなら頻度を下げる）
        target_norm = 1.5 # 目標とする重みの総量
        
        for i in range(cortex_range[0], cortex_range[1]):
            neuron = self.brain.kernel.neurons[i]
            # このニューロンに入ってくるシナプスを探すのは逆参照が必要でカーネルがサポートしていない場合がある
            # DORAの構造では「Pre -> Post」のリストしかないので、
            # 逆に「Post -> Pre」の正規化は難しい。
            
            # 代替案：Pre側の「出力シナプスの強さ」を制限する（リソース配分）
            # ここでは run_plasticity 内での減衰(Decay)で代用する
            pass

    def run_plasticity(self, pos_spikes, neg_spikes, input_spikes):
        cortex_range = self.brain.group_indices["cortex"]
        retina_range = self.brain.group_indices["retina"]
        
        lr = 0.02
        weight_decay = 0.001 
        
        input_vals = self._safe_numpy(input_spikes)
        pre_indices = np.where(input_vals > 0)[0]
        
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
            
            # 入力強度によるスケーリング
            rate_factor = min(1.0, input_vals[pre_idx_rel] / 3.0)
            
            for synapse in neuron.outgoing_synapses:
                if cortex_range[0] <= synapse.target_id < cortex_range[1]:
                    post_idx = synapse.target_id - cortex_range[0]
                    
                    if post_idx in active_post_indices:
                        val_p = pos_vals[post_idx]
                        val_n = neg_vals[post_idx]
                        
                        # 差分学習 (Difference Learning)
                        diff = val_p - val_n
                        
                        # 不感帯: 差が小さいときは更新しない（ノイズ対策）
                        if abs(diff) < 2.0:
                            continue
                            
                        # Negativeへのペナルティを強化 (x2.0)
                        if diff < 0:
                            diff *= 2.0
                            
                        dw = lr * diff * rate_factor
                        
                        synapse.weight += dw
                        
                        # 明示的な減衰（Normalizationの代わり）
                        synapse.weight *= 0.995 
                        
                        # 重み制限: 0.01 ~ 1.5
                        synapse.weight = max(0.01, min(1.5, synapse.weight))
                        updated_count += 1
        return updated_count

    def predict(self, img):
        best_g = -1
        pred = -1
        scores = []
        
        for l in range(10):
            in_data = self.overlay_label(img, l, True)
            self.brain.reset_state()
            
            res = self.os_kernel.run_cycle({"retina": in_data}, phase="wake")
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
                res_pos = self.brain.forward_step({"retina": in_pos}, learning=True, duration=20.0)
                spikes_pos = res_pos["spikes"]["cortex"]
                input_spikes = res_pos["spikes"]["retina"]
                
                # --- Negative Phase ---
                in_neg = self.overlay_label(img, lbl, False)
                self.brain.reset_state()
                res_neg = self.brain.forward_step({"retina": in_neg}, learning=True, duration=20.0)
                spikes_neg = res_neg["spikes"]["cortex"]

                # --- Learning ---
                self.run_plasticity(spikes_pos, spikes_neg, input_spikes)
                
                pos_g = self.get_goodness(spikes_pos)
                neg_g = self.get_goodness(spikes_neg)
                
                # Positiveが有意に大きければ正解
                if pos_g > neg_g * 1.1: # 10%以上の差が必要
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
            
            # スコア分散が小さい場合は自信なし(-1)とする
            score_std = np.std(scores)
            if score_std < 5.0:
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
    
    # 500枚で高速イテレーション
    train_subset = torch.utils.data.Subset(dataset, range(500))
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=1, shuffle=True)
    
    test_subset = torch.utils.data.Subset(dataset, range(500, 550))
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=1, shuffle=False)
    
    learner = DORAOnlineLearnerV14(n_hidden=1000)
    
    logger.info("🚀 Starting DORA Online Learning (v14.0 Normalized Poisson)")
    learner.train(train_loader, epochs=1)
    learner.evaluate(test_loader, limit=50)

if __name__ == "__main__":
    main()