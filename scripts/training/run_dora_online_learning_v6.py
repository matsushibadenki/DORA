# scripts/training/run_dora_online_learning_v6.py
# Japanese Title: DORA オンライン学習 v13.0 (ポアソン・レートコーディング版)
# Description: 入力画像を時間的なスパイク列(Poisson Spike Train)に変換して入力することで、信号強度をアナログ的に表現し、学習の安定性と精度を飛躍させる。

import sys
import os
import logging
import torch
import numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm

sys.path.append(os.getcwd())

# 標準のSNNモジュールを使用（パッチなし）
from snn_research.core.snn_core import SpikingNeuralSubstrate
from snn_research.core.neuromorphic_os import NeuromorphicOS

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DORA_Learner_v13_Poisson")

# -----------------------------------------------------------------------------
# 【拡張】ポアソン入力対応SNNコア
# -----------------------------------------------------------------------------
class PoissonSNN(SpikingNeuralSubstrate):
    def forward_step(self, ext_inputs: dict, learning: bool = True, dreaming: bool = False, **kwargs) -> dict:
        """
        画像をポアソンスパイク列に変換してシミュレーションを実行するオーバーライドメソッド
        """
        simulation_duration = kwargs.get("duration", 50.0) # 1枚あたり50ms処理
        
        # 1. 入力スパイクの生成 (Poisson Process)
        if not dreaming:
            for name, tensor in ext_inputs.items():
                if name in self.group_indices:
                    start_id, _ = self.group_indices[name]
                    # テンソルを確率密度として扱う (0.0~1.0)
                    # 値が大きいほど、高頻度でスパイクが発生する
                    input_probs = torch.clamp(tensor.flatten(), 0, 1).cpu().numpy()
                    
                    # アクティブな画素について、時間軸上でスパイクを生成
                    active_indices = np.where(input_probs > 0.1)[0]
                    
                    for idx in active_indices:
                        rate = input_probs[idx] * 0.1 # スパイク生成確率係数
                        # durationの間、毎ステップ確率判定
                        for t in range(int(simulation_duration)):
                            if np.random.random() < rate:
                                self.kernel.push_input_spikes([int(start_id + idx)], self.kernel.current_time + t + 0.1)

        # 2. カーネル実行
        counts = self.kernel.run(duration=simulation_duration, learning_enabled=learning)
        
        # 3. 結果の集計
        curr_spikes = {}
        for name, (s, e) in self.group_indices.items():
            spikes = torch.zeros(1, e-s, device=self.device)
            for nid, count in counts.items():
                if s <= nid < e and count > 0:
                    spikes[0, nid-s] = count # スパイク回数を記録（強さになる）
            curr_spikes[name] = spikes
            
        self.uncertainty_score = 0.0 # 簡易化
        return {"spikes": curr_spikes}

# -----------------------------------------------------------------------------

class DORAOnlineLearnerV13:
    def __init__(self, n_hidden=1000, device='cpu'):
        self.device = torch.device(device)
        self.n_hidden = n_hidden
        
        self.config = {
            "dt": 1.0, 
            "t_ref": 2.0,   # 不応期
            "tau_m": 20.0,  # 膜時定数（標準的な減衰あり）
        }
        # 拡張したSNNクラスを使用
        self.brain = PoissonSNN(self.config, device=self.device)
        
        # 1. ニューロン定義
        self.brain.add_neuron_group("retina", 794, v_thresh=0.5)
        self.brain.add_neuron_group("cortex", n_hidden, v_thresh=1.0)
        
        # 2. 接続構築
        logger.info(f"🔗 Building connections (Hidden={n_hidden})...")
        retina_range = self.brain.group_indices["retina"]
        cortex_range = self.brain.group_indices["cortex"]
        
        n_input = retina_range[1] - retina_range[0]
        n_cortex = cortex_range[1] - cortex_range[0]
        
        # 重み設定: レートコーディング用
        # 何度もスパイクが来るので、重みは小さくて良い
        weights = np.random.uniform(0.02, 0.05, (n_cortex, n_input))
        
        # ラベルは強く (確実なガイド)
        label_start_idx = 784
        weights[:, label_start_idx:] = 2.0 
        
        # 接続密度 (15%)
        mask = (np.random.random(weights.shape) < 0.15).astype(float)
        weights *= mask
        weights[:, label_start_idx:] = 2.0
        
        self.brain.kernel.connect_groups(retina_range, cortex_range, weights)
        
        # 側抑制 (-1.0)
        inhibition_weights = -1.0 * np.ones((n_cortex, n_cortex))
        np.fill_diagonal(inhibition_weights, 0)
        inhib_mask = (np.random.random(inhibition_weights.shape) < 0.20).astype(float)
        inhibition_weights *= inhib_mask
        
        self.brain.kernel.connect_groups(cortex_range, cortex_range, inhibition_weights)
        
        self.brain._projections_registry["optic_nerve"] = {"source": "retina", "target": "cortex"}
        self.brain._projections_registry["lateral_inhibition"] = {"source": "cortex", "target": "cortex"}
        
        # 構造的可塑性はオフ（安定化のため）
        self.brain.kernel.structural_plasticity_enabled = False
        
        # 3. OS起動
        self.os_kernel = NeuromorphicOS(self.brain, tick_rate=50)
        self.os_kernel.boot()
        
        logger.info("🧠 Brain Initialized: Poisson Rate Coding Mode (Duration=50ms).")

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
        
        lr = 0.01 # レートコーディングなので学習率は控えめに
        weight_decay = 0.0005 
        
        # 入力スパイク数が多い順に処理（効率化）
        input_vals = self._safe_numpy(input_spikes) # 今回はretinaのスパイク数
        pre_indices = np.where(input_vals > 0)[0]
        
        pos_vals = self._safe_numpy(pos_spikes)
        neg_vals = self._safe_numpy(neg_spikes)
        
        updated_count = 0
        active_post_indices = np.where((pos_vals > 0) | (neg_vals > 0))[0]
        
        if len(active_post_indices) == 0:
            return 0

        # ベクトル化したいが、SNNの構造上ループで処理
        for pre_idx_rel in pre_indices:
            pre_id = retina_range[0] + pre_idx_rel
            if pre_id >= len(self.brain.kernel.neurons): continue
            
            neuron = self.brain.kernel.neurons[pre_id]
            
            # 入力頻度に応じたスケーリング
            rate_factor = min(1.0, input_vals[pre_idx_rel] / 5.0) 
            
            for synapse in neuron.outgoing_synapses:
                if cortex_range[0] <= synapse.target_id < cortex_range[1]:
                    post_idx = synapse.target_id - cortex_range[0]
                    
                    if post_idx in active_post_indices:
                        val_p = pos_vals[post_idx]
                        val_n = neg_vals[post_idx]
                        
                        # Forward-Forward則 (レートベース)
                        # よく発火する入力に対して感度を上げる
                        dw = lr * (val_p - val_n) * rate_factor
                        
                        synapse.weight += dw
                        synapse.weight -= weight_decay * synapse.weight
                        synapse.weight = max(0.001, min(1.0, synapse.weight))
                        updated_count += 1
        return updated_count

    def predict(self, img):
        best_g = -1
        pred = -1
        scores = []
        
        for l in range(10):
            in_data = self.overlay_label(img, l, True)
            self.brain.reset_state()
            
            # durationを引数で渡す (PoissonSNNで処理される)
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
                # 50msかけてじっくり処理
                res_pos = self.brain.forward_step({"retina": in_pos}, learning=True, duration=50.0)
                spikes_pos = res_pos["spikes"]["cortex"]
                input_spikes = res_pos["spikes"]["retina"] # retinaの発火数も返ってくる
                
                # --- Negative Phase ---
                in_neg = self.overlay_label(img, lbl, False)
                self.brain.reset_state()
                res_neg = self.brain.forward_step({"retina": in_neg}, learning=True, duration=50.0)
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
    
    # 時間がかかるのでデータ数を減らす
    train_subset = torch.utils.data.Subset(dataset, range(500))
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=1, shuffle=True)
    
    test_subset = torch.utils.data.Subset(dataset, range(500, 550))
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=1, shuffle=False)
    
    learner = DORAOnlineLearnerV13(n_hidden=1000)
    
    logger.info("🚀 Starting DORA Online Learning (v13.0 Poisson Rate Coding)")
    learner.train(train_loader, epochs=1)
    learner.evaluate(test_loader, limit=50)

if __name__ == "__main__":
    main()