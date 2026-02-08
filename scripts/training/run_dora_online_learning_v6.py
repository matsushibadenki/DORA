# scripts/training/run_dora_online_learning_v6.py
# Japanese Title: DORA オンライン学習 v16.0 (ポアソン発火率・強度整合版)
# Description: 理論計算に基づき、入力スパイクの頻度と重みの積が閾値を確実に超えるよう再設計。沈黙を打破し、教師あり学習を成立させる。

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
logger = logging.getLogger("DORA_Learner_v16_TunedPoisson")

# -----------------------------------------------------------------------------
# ポアソン入力対応SNNコア (レート調整版)
# -----------------------------------------------------------------------------
class PoissonSNN(SpikingNeuralSubstrate):
    def forward_step(self, ext_inputs: dict, learning: bool = True, dreaming: bool = False, **kwargs) -> dict:
        simulation_duration = kwargs.get("duration", 30.0) # 30ms
        
        if not dreaming:
            for name, tensor in ext_inputs.items():
                if name in self.group_indices:
                    start_id, _ = self.group_indices[name]
                    input_probs = torch.clamp(tensor.flatten(), 0, 1).cpu().numpy()
                    
                    active_indices = np.where(input_probs > 0.2)[0]
                    
                    for idx in active_indices:
                        # 【修正1】レート係数を 0.1 -> 0.2 に上げ、スパイク密度を高める
                        # 確率1.0の画素は、平均して5ステップに1回発火する
                        rate = input_probs[idx] * 0.2 
                        
                        # 確率的スパイク生成
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

class DORAOnlineLearnerV16:
    def __init__(self, n_hidden=1000, device='cpu'):
        self.device = torch.device(device)
        self.n_hidden = n_hidden
        
        self.config = {
            "dt": 1.0, 
            "t_ref": 1.0,   # 【修正2】不応期を短くして高頻度発火を許容
            "tau_m": 20.0,
        }
        self.brain = PoissonSNN(self.config, device=self.device)
        
        # 1. ニューロン定義
        self.brain.add_neuron_group("retina", 794, v_thresh=0.5)
        
        # 閾値 5.0
        self.brain.add_neuron_group("cortex", n_hidden, v_thresh=5.0)
        
        # 2. 接続構築
        logger.info(f"🔗 Building connections (Hidden={n_hidden})...")
        retina_range = self.brain.group_indices["retina"]
        cortex_range = self.brain.group_indices["cortex"]
        
        n_input = retina_range[1] - retina_range[0]
        n_cortex = cortex_range[1] - cortex_range[0]
        
        # 【修正3】画像入力の重みを強化 (0.01 -> 0.1)
        # 計算: 15画素(接続) * 6スパイク(30ms/5) * 0.1(重み) = 9.0 (最大値)
        # 平均的には 3.0 ~ 4.0 程度になり、閾値5.0には届かないが、ラベルがあれば超える
        weights = np.random.uniform(0.08, 0.12, (n_cortex, n_input))
        
        # 【修正4】ラベル入力 (3.0)
        # ラベルからのスパイク(約6回) * 3.0 = 18.0 (圧倒的)
        # ただしスパイクタイミングがばらつくので、瞬間的な電位寄与はもっと低い
        # 実際には「ラベルがある＝発火確定」という強いバイアスになる
        label_start_idx = 784
        weights[:, label_start_idx:] = 3.0 
        
        # 接続密度 (10%)
        mask = (np.random.random(weights.shape) < 0.10).astype(float)
        weights *= mask
        weights[:, label_start_idx:] = 3.0 
        
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
        
        self.os_kernel = NeuromorphicOS(self.brain, tick_rate=50)
        self.os_kernel.boot()
        
        logger.info("🧠 Brain Initialized: Tuned Rate Coding (Im~0.1, Lb~3.0).")

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

    def run_plasticity(self, pos_spikes, neg_spikes, input_spikes):
        cortex_range = self.brain.group_indices["cortex"]
        retina_range = self.brain.group_indices["retina"]
        
        lr = 0.05 
        weight_decay = 0.0002
        
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
            rate_factor = min(2.0, input_vals[pre_idx_rel] / 5.0)
            
            for synapse in neuron.outgoing_synapses:
                if cortex_range[0] <= synapse.target_id < cortex_range[1]:
                    post_idx = synapse.target_id - cortex_range[0]
                    
                    if post_idx in active_post_indices:
                        val_p = pos_vals[post_idx]
                        val_n = neg_vals[post_idx]
                        
                        diff = val_p - val_n
                        
                        # Negativeペナルティ (間違いを減らす)
                        if diff < 0:
                            diff *= 2.0 
                            
                        dw = lr * diff * rate_factor
                        
                        synapse.weight += dw
                        synapse.weight *= 0.998 # 減衰
                        synapse.weight = max(0.001, min(1.0, synapse.weight)) # 上限1.0
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
                res_pos = self.brain.forward_step({"retina": in_pos}, learning=True, duration=30.0)
                spikes_pos = res_pos["spikes"]["cortex"]
                input_spikes = res_pos["spikes"]["retina"]
                
                # --- Negative Phase ---
                in_neg = self.overlay_label(img, lbl, False)
                self.brain.reset_state()
                res_neg = self.brain.forward_step({"retina": in_neg}, learning=True, duration=30.0)
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
    
    # 500枚
    train_subset = torch.utils.data.Subset(dataset, range(500))
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=1, shuffle=True)
    
    test_subset = torch.utils.data.Subset(dataset, range(500, 550))
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=1, shuffle=False)
    
    learner = DORAOnlineLearnerV16(n_hidden=1000)
    
    logger.info("🚀 Starting DORA Online Learning (v16.0 Tuned Poisson)")
    learner.train(train_loader, epochs=1)
    learner.evaluate(test_loader, limit=50)

if __name__ == "__main__":
    main()