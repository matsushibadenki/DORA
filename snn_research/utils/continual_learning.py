# snn_research/utils/continual_learning.py
# Title: Continual Learning Utilities
# Description:
#   継続学習デモで使用する共通コンポーネント（リプレイバッファ、特徴抽出器）の定義。

import torch
import torch.nn as nn
import random
from typing import List, Tuple, Optional
from tqdm import tqdm
from snn_research.core.ensemble_scal import EnsembleSCAL

class SCALFeatureExtractor(nn.Module):
    """SCALを用いた凍結可能な特徴抽出器"""
    def __init__(self, in_features, out_features, n_models=5, device='cpu'):
        super().__init__()
        self.scal = EnsembleSCAL(
            in_features=in_features, out_features=out_features, n_models=n_models).to(device)
        self.device = device
        self.norm = nn.LayerNorm(out_features).to(device)

    def forward(self, x):
        if not x.is_contiguous():
            x = x.contiguous()
        x_flat = x.view(x.size(0), -1).to(self.device)
        with torch.no_grad():
            out = self.scal(x_flat)
        return self.norm(out['output'])

    def fit(self, data_loader, epochs=1):
        self.scal.train()
        print(f"  SCAL training ({epochs} epochs)...")
        for _ in range(epochs):
            pbar = tqdm(data_loader, desc="SCAL Fitting")
            for data, _ in pbar:
                if not data.is_contiguous():
                    data = data.contiguous()
                data = data.view(data.size(0), -1).to(self.device)
                self.scal(data)

class HippocampalReplayBuffer:
    """海馬リプレイバッファ（経験再生用）"""
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.seen_count = 0

    def push(self, features, label):
        features_cpu = features.detach().cpu()
        label_cpu = label.detach().cpu()

        batch_size = features.size(0)

        for i in range(batch_size):
            item = (features_cpu[i].clone(), label_cpu[i].clone())

            if len(self.buffer) < self.capacity:
                self.buffer.append(item)
            else:
                m = random.randint(0, self.seen_count)
                if m < self.capacity:
                    self.buffer[m] = item

            self.seen_count += 1

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None, None

        batch = random.sample(self.buffer, batch_size)
        features, labels = zip(*batch)
        return torch.stack(features), torch.stack(labels)

    def clear(self):
        self.buffer.clear()

    def __len__(self): return len(self.buffer)