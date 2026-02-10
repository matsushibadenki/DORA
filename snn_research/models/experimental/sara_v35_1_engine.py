# directory: snn_research/models/experimental/sara_v35_1_engine.py
# title: SARA Engine v35.1 - Liquid Harmony
# description: Fast層の暴走を抑え、適応的Homeostasisと不応期調整を導入した、95%を目指すためのチューニング版LSM。

import numpy as np
import random
from typing import List, Tuple, Dict, Optional

class TrueLiquidLayer:
    def __init__(self, input_size: int, hidden_size: int, decay: float, input_scale: float, rec_scale: float, density: float = 0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.decay = decay
        
        # --- Input Weights ---
        self.in_indices = []
        self.in_weights = []
        
        fan_in = int(input_size * density)
        if fan_in == 0: fan_in = 1
        w_range_in = input_scale * np.sqrt(3.0 / fan_in)
        
        for i in range(input_size):
            n = int(hidden_size * density)
            if n > 0:
                idx = np.random.choice(hidden_size, n, replace=False).astype(np.int32)
                w = np.random.uniform(-w_range_in, w_range_in, n).astype(np.float32)
                self.in_indices.append(idx)
                self.in_weights.append(w)
            else:
                self.in_indices.append(np.array([], dtype=np.int32))
                self.in_weights.append(np.array([], dtype=np.float32))

        # --- Recurrent Weights ---
        self.rec_indices = []
        self.rec_weights = []
        
        rec_density = 0.1
        fan_in_rec = int(hidden_size * rec_density)
        if fan_in_rec == 0: fan_in_rec = 1
        w_range_rec = rec_scale / np.sqrt(fan_in_rec)
        
        print(f"  - Liquid Layer: Size={hidden_size}, Decay={decay:.2f}, "
              f"InputScale={input_scale:.2f}, RecScale={rec_scale:.2f}")
        
        for i in range(hidden_size):
            n = int(hidden_size * rec_density)
            if n > 0:
                idx = np.random.choice(hidden_size, n, replace=False).astype(np.int32)
                w = np.random.uniform(-w_range_rec, w_range_rec, n).astype(np.float32)
                self.rec_indices.append(idx)
                self.rec_weights.append(w)
            else:
                self.rec_indices.append(np.array([], dtype=np.int32))
                self.rec_weights.append(np.array([], dtype=np.float32))

        # State
        self.v = np.zeros(hidden_size, dtype=np.float32)
        self.refractory = np.zeros(hidden_size, dtype=np.float32)
        
        # --- Tuned Homeostasis Params ---
        # Fast層のTarget Rateを下げ、閾値を上げる
        if decay < 0.4:  # Fast
            self.base_thresh = 0.8  # 0.5 -> 0.8
            self.target_rate = 0.02  # 0.10 -> 0.02 (重要: 暴走抑制)
            self.refractory_period = 3.0 # 不応期を長く
        elif decay < 0.8:  # Medium
            self.base_thresh = 0.8
            self.target_rate = 0.03  # 0.05 -> 0.03
            self.refractory_period = 2.0
        else:  # Slow
            self.base_thresh = 1.0
            self.target_rate = 0.02  # 0.03 -> 0.02
            self.refractory_period = 1.5
            
        self.thresh = np.ones(hidden_size, dtype=np.float32) * self.base_thresh

    def reset(self):
        self.v.fill(0)
        self.refractory.fill(0)

    def update_homeostasis(self, activity_history: np.ndarray, steps: int):
        """Adaptive Gain Homeostasis"""
        rate = activity_history / float(steps)
        diff = rate - self.target_rate
        
        # 乖離が大きい場合は強く補正する (Adaptive Gain)
        gain = np.where(np.abs(diff) > 0.05, 0.15, 0.03)
        
        self.thresh += gain * diff
        self.thresh = np.clip(self.thresh, self.base_thresh * 0.5, self.base_thresh * 5.0)

    def forward(self, active_inputs: List[int], prev_active_hidden: List[int]) -> List[int]:
        # Refractory Decay
        self.refractory = np.maximum(0, self.refractory - 1)
        
        # Voltage Decay
        self.v *= self.decay
        
        # External Input
        for pre_id in active_inputs:
            if pre_id < len(self.in_indices):
                targets = self.in_indices[pre_id]
                ws = self.in_weights[pre_id]
                if len(targets) > 0:
                    self.v[targets] += ws
                    
        # Recurrent Input
        for pre_h_id in prev_active_hidden:
            if pre_h_id < len(self.rec_indices):
                targets = self.rec_indices[pre_h_id]
                ws = self.rec_weights[pre_h_id]
                if len(targets) > 0:
                    self.v[targets] += ws
        
        # Fire
        ready_mask = (self.v >= self.thresh) & (self.refractory <= 0)
        fired_indices = np.where(ready_mask)[0]
        
        if len(fired_indices) > 0:
            # Soft Reset
            self.v[fired_indices] -= self.thresh[fired_indices]
            self.v = np.maximum(self.v, 0.0)
            
            # Adaptive Refractory
            self.refractory[fired_indices] = self.refractory_period
            
        return fired_indices.tolist()

class SaraEngineV35_1:
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        
        # Recurrent Scale Tuning: Fast層の再帰を少し強める
        self.reservoirs = [
            TrueLiquidLayer(input_size, 1500, decay=0.3, input_scale=1.0, rec_scale=1.2),  # Fast
            TrueLiquidLayer(input_size, 2000, decay=0.7, input_scale=0.8, rec_scale=1.5),  # Med
            TrueLiquidLayer(input_size, 1500, decay=0.95, input_scale=0.4, rec_scale=2.0), # Slow
        ]
        
        self.total_hidden = sum(r.hidden_size for r in self.reservoirs)
        self.offsets = [0, 1500, 3500]
        
        # Readout (Xavier)
        self.w_ho = []
        self.m_ho = []
        self.v_ho = []
        
        for _ in range(output_size):
            limit = np.sqrt(6.0 / (self.total_hidden + output_size))
            w = np.random.uniform(-limit, limit, self.total_hidden).astype(np.float32)
            self.w_ho.append(w)
            self.m_ho.append(np.zeros(self.total_hidden, dtype=np.float32))
            self.v_ho.append(np.zeros(self.total_hidden, dtype=np.float32))
            
        self.o_v = np.zeros(output_size, dtype=np.float32)
        
        # Adam
        self.lr = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.o_decay = 0.9
        
        self.layer_activity_counters = [np.zeros(r.hidden_size, dtype=np.float32) for r in self.reservoirs]
        self.prev_spikes = [[] for _ in self.reservoirs]
        
        print(f"Total Liquid Neurons: {self.total_hidden}")

    def reset_state(self):
        for r in self.reservoirs:
            r.reset()
        self.o_v.fill(0)
        for c in self.layer_activity_counters:
            c.fill(0)
        self.prev_spikes = [[] for _ in self.reservoirs]

    def sleep_phase(self, prune_rate: float = 0.05):
        """Adaptive Pruning"""
        print(f"  [Sleep Phase] Adaptive Pruning...")
        pruned_total = 0
        total_weights = 0
        
        for o in range(self.output_size):
            weights = self.w_ho[o]
            total_weights += len(weights)
            
            weights *= 0.995 # Weight Decay
            
            abs_w = np.abs(weights)
            nonzero_w = abs_w[abs_w > 1e-6]
            
            if len(nonzero_w) > 0:
                threshold = np.percentile(nonzero_w, prune_rate * 100)
                mask = abs_w < threshold
                pruned_total += np.sum(mask)
                weights[mask] = 0.0
            
            norm = np.linalg.norm(weights)
            if norm > 5.0:
                weights *= (5.0 / norm)
                
            self.w_ho[o] = weights
            
        print(f"  [Sleep Phase] Pruned {pruned_total} / {total_weights} weights ({pruned_total/total_weights*100:.2f}%)")

    def train_step(self, spike_train: List[List[int]], target_label: int, dropout_rate: float = 0.1):
        self.reset_state()
        grad_accumulator = [np.zeros_like(w) for w in self.w_ho]
        steps = len(spike_train)
        
        for input_spikes in spike_train:
            # Dropout (Fast stochastic drop)
            if dropout_rate > 0.0 and len(input_spikes) > 2:
                # Randomly drop inputs
                # Using a mask is faster than list comprehension if inputs are numpy array, 
                # but input_spikes is list. 
                # Let's use simple probability check per list (fast enough for small lists)
                if random.random() < 0.5: # 50% chance to apply dropout at all
                     active_inputs = [idx for idx in input_spikes if random.random() > dropout_rate]
                else:
                     active_inputs = input_spikes
            else:
                active_inputs = input_spikes

            # Forward
            all_hidden_spikes = []
            
            for i, r in enumerate(self.reservoirs):
                local_spikes = r.forward(active_inputs, self.prev_spikes[i])
                self.prev_spikes[i] = local_spikes
                
                if local_spikes:
                    self.layer_activity_counters[i][local_spikes] += 1.0
                    base = self.offsets[i]
                    all_hidden_spikes.extend([idx + base for idx in local_spikes])
            
            self.o_v *= self.o_decay
            
            if not all_hidden_spikes:
                continue

            # Readout
            num_spikes = len(all_hidden_spikes)
            scale_factor = 10.0 / (num_spikes + 20.0)
            
            for o in range(self.output_size):
                current = np.sum(self.w_ho[o][all_hidden_spikes])
                self.o_v[o] += current * scale_factor
            
            if np.max(self.o_v) > 0:
                self.o_v -= 0.1 * np.mean(self.o_v)
            self.o_v = np.clip(self.o_v, -5.0, 5.0)

            # Error
            errors = np.zeros(self.output_size, dtype=np.float32)
            if self.o_v[target_label] < 1.0:
                errors[target_label] = 1.0 - self.o_v[target_label]
            
            for o in range(self.output_size):
                if o != target_label and self.o_v[o] > -0.1:
                    errors[o] = -0.1 - self.o_v[o]
            
            for o in range(self.output_size):
                if abs(errors[o]) > 0.01:
                    grad_accumulator[o][all_hidden_spikes] += errors[o]

        # Adam Update
        for o in range(self.output_size):
            grad = grad_accumulator[o]
            self.m_ho[o] = self.beta1 * self.m_ho[o] + (1 - self.beta1) * grad
            self.v_ho[o] = self.beta2 * self.v_ho[o] + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m_ho[o]
            v_hat = self.v_ho[o]
            self.w_ho[o] += self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            np.clip(self.w_ho[o], -3.0, 3.0, out=self.w_ho[o])
        
        # Homeostasis Update
        for i, r in enumerate(self.reservoirs):
            r.update_homeostasis(self.layer_activity_counters[i], steps)

    def predict(self, spike_train: List[List[int]]) -> int:
        self.reset_state()
        total_potentials = np.zeros(self.output_size, dtype=np.float32)
        
        for input_spikes in spike_train:
            all_hidden_spikes = []
            for i, r in enumerate(self.reservoirs):
                local = r.forward(input_spikes, self.prev_spikes[i])
                self.prev_spikes[i] = local
                base = self.offsets[i]
                all_hidden_spikes.extend([x + base for x in local])
            
            self.o_v *= self.o_decay
            if all_hidden_spikes:
                num_spikes = len(all_hidden_spikes)
                scale_factor = 10.0 / (num_spikes + 20.0)
                for o in range(self.output_size):
                    self.o_v[o] += np.sum(self.w_ho[o][all_hidden_spikes]) * scale_factor
            
            if np.max(self.o_v) > 0:
                self.o_v -= 0.1 * np.mean(self.o_v)
            total_potentials += self.o_v
            
        return int(np.argmax(total_potentials))
    
    def get_activity_stats(self, spike_train):
        self.reset_state()
        counts = [0, 0, 0]
        for input_spikes in spike_train:
            for i, r in enumerate(self.reservoirs):
                s = r.forward(input_spikes, self.prev_spikes[i])
                self.prev_spikes[i] = s
                counts[i] += len(s)
        return [c / len(spike_train) for c in counts]