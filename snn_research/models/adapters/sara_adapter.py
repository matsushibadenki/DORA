# directory: snn_research/models/adapters
# file: sara_adapter.py
# purpose: SARAエンジン アダプター v4 (With Experience Replay for Stability)

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import random
from PIL import Image
from torchvision import transforms
from typing import Tuple, Optional, Any, List

try:
    from snn_research.models.experimental.sara_engine import SARAEngine
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
    from snn_research.models.experimental.sara_engine import SARAEngine

class SARAAdapter:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = SARAEngine(
            input_dim=784,
            n_encode_neurons=128,
            d_legendre=64,
            d_meaning=128,
            n_output=10
        ).to(self.device)
        
        self._load_weights(model_path)
        self.model.eval()
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        
        # 短期記憶バッファ (海馬)
        self.episodic_memory: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.memory_capacity = 100
        
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def _load_weights(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model weights not found at {path}")
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"[SARA] Loaded weights from {path}")
        except Exception as e:
            print(f"[SARA] Error loading weights: {e}")

    def think(self, image_input: Any) -> dict:
        self.model.eval()
        img_tensor = self._preprocess(image_input)
        with torch.no_grad():
            logits, rate, _ = self.model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, prediction = torch.max(probs, dim=1)
        return {
            "prediction": prediction.item(),
            "confidence": confidence.item(),
            "firing_rate": rate
        }

    def learn_instance(self, image_input: Any, correct_label: int, max_steps: int = 50) -> float:
        """
        即時学習 + Experience Replay (リハーサル)
        新しい情報を学ぶ際に、短期記憶からランダムに過去の事例を混ぜて学習し、
        特定のパターンへの過剰適合（破滅的忘却）を防ぐ。
        """
        self.model.train()
        
        # 学習率を少し控えめに調整 (過学習防止)
        current_lr = 0.02
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
            
        new_img = self._preprocess(image_input)
        new_target = torch.tensor([correct_label], device=self.device)
        
        # 海馬へ保存
        if len(self.episodic_memory) >= self.memory_capacity:
            self.episodic_memory.pop(0)
        # CPUに退避して保存
        self.episodic_memory.append((new_img.detach().cpu(), new_target.detach().cpu()))
        
        initial_loss = 0.0
        print(f"[Plasticity] Adapting with Replay...", end="", flush=True)
        
        for step in range(max_steps):
            self.optimizer.zero_grad()
            
            # --- Replay Batch 生成 ---
            # 今回の新しいデータ
            batch_imgs = [new_img]
            batch_targets = [new_target]
            
            # 過去の記憶からランダムに数件取得 (Replay)
            # 記憶が十分ある場合、バッチサイズを増やして安定化
            replay_count = min(len(self.episodic_memory) - 1, 4) 
            if replay_count > 0:
                replay_samples = random.sample(self.episodic_memory[:-1], replay_count) # 最新以外からサンプリング
                for img_cpu, target_cpu in replay_samples:
                    batch_imgs.append(img_cpu.to(self.device))
                    batch_targets.append(target_cpu.to(self.device))
            
            # バッチ結合
            input_batch = torch.cat(batch_imgs, dim=0)
            target_batch = torch.cat(batch_targets, dim=0)
            
            # --- 学習ステップ ---
            logits, _, _ = self.model(input_batch)
            loss = nn.functional.cross_entropy(logits, target_batch)
            
            if step == 0: initial_loss = loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # --- 収束判定 (ターゲット画像に対してのみ行う) ---
            with torch.no_grad():
                logits_new, _, _ = self.model(new_img)
                probs = torch.softmax(logits_new, dim=1)
                pred = probs.argmax(dim=1).item()
                confidence = probs[0, correct_label].item()
            
            # 確信度が十分高くなったら終了
            if pred == correct_label and confidence > 0.90:
                print(f" Converged @ {step+1} (Conf: {confidence*100:.1f}%)")
                break
        else:
            print(" Stopped.")
            
        self.model.eval()
        return initial_loss

    def sleep(self, epochs: int = 3):
        """睡眠統合 (全エピソード記憶のリプレイ)"""
        if not self.episodic_memory:
            print("[Sleep] No memories to consolidate.")
            return
            
        print(f"\n[Sleep] Consolidating {len(self.episodic_memory)} episodes...")
        self.model.train()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 0.001 # 低学習率
            
        total_loss = 0
        for epoch in range(epochs):
            random.shuffle(self.episodic_memory)
            # ミニバッチ学習の簡易実装
            for img_cpu, target_cpu in self.episodic_memory:
                img = img_cpu.to(self.device)
                target = target_cpu.to(self.device)
                
                self.optimizer.zero_grad()
                logits, _, _ = self.model(img)
                loss = nn.functional.cross_entropy(logits, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()
                
        print(f"[Sleep] Wake up. (Avg Loss: {total_loss / (epochs * len(self.episodic_memory)):.4f})")
        self.episodic_memory.clear()
        self.model.eval()

    def _preprocess(self, image_input: Any) -> torch.Tensor:
        img = None
        if isinstance(image_input, str): img = Image.open(image_input).convert('L')
        elif isinstance(image_input, np.ndarray): img = Image.fromarray(image_input).convert('L')
        elif isinstance(image_input, Image.Image): img = image_input.convert('L')
        elif isinstance(image_input, torch.Tensor):
            if image_input.dim() == 2: img = transforms.ToPILImage()(image_input)
            elif image_input.dim() == 3: img = transforms.ToPILImage()(image_input)
            else: return image_input.view(1, -1).to(self.device)
        else: raise ValueError(f"Unsupported type: {type(image_input)}")
        return self.transform(img).view(1, -1).to(self.device)