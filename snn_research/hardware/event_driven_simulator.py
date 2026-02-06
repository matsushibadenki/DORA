# snn_research/hardware/event_driven_simulator.py
# Title: DORA Kernel v1.7 (Compat Alias)
# Description: EventDrivenSimulatorエイリアスを追加し、旧スクリプトとの互換性を確保。

import heapq
import logging
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

# --- Data Structures ---

@dataclass(order=True)
class SpikeEvent:
    timestamp: float
    neuron_id: int = field(compare=False)
    payload: float = field(compare=False, default=1.0)

class Synapse:
    __slots__ = ('target_id', 'weight', 'trace', 'delay')
    def __init__(self, target_id: int, weight: float, delay: float = 1.0):
        self.target_id = target_id
        self.weight = weight
        self.delay = delay
        self.trace = 0.0

class NeuronNode:
    __slots__ = (
        'id', 'layer_id', 'v', 'v_thresh', 'v_reset', 'tau_mem', 
        'last_spike_time', 'prediction_error', 'outgoing_synapses',
        'refractory_period', 'is_inhibitory'
    )
    def __init__(self, neuron_id: int, layer_id: int, 
                 v_thresh: float = 0.5, tau_mem: float = 50.0,
                 refractory_period: float = 3.0, is_inhibitory: bool = False):
        self.id = neuron_id
        self.layer_id = layer_id
        self.v = 0.0
        self.v_thresh = v_thresh
        self.v_reset = 0.0
        self.tau_mem = tau_mem
        self.last_spike_time = -100.0
        self.refractory_period = refractory_period
        self.is_inhibitory = is_inhibitory
        self.prediction_error = 0.0
        self.outgoing_synapses: List[Synapse] = []

    def integrate(self, weight: float, dt: float) -> None:
        decay = math.exp(-dt / self.tau_mem)
        self.v = self.v * decay + weight
        if self.v < -5.0: self.v = -5.0

    def check_fire(self, current_time: float) -> bool:
        if (current_time - self.last_spike_time) < self.refractory_period:
            return False
        if self.v >= self.v_thresh:
            self.prediction_error = self.v - self.v_thresh
            self.v = self.v_reset
            self.last_spike_time = current_time
            return True
        return False

# --- Kernel ---

class DORAKernel:
    def __init__(self, dt: float = 1.0):
        self.neurons: List[NeuronNode] = []
        self.event_queue: List[SpikeEvent] = []
        self.current_time = 0.0
        self.dt = dt
        self.stats = {"ops": 0, "spikes": 0, "plasticity_events": 0}
        self.spike_history: List[Tuple[float, int, bool]] = []
        logger.info("🧠 DORA Kernel v1.7 (Compat Alias) initialized")

    def add_neuron(self, layer_id: int = 0, v_thresh: float = 0.5, tau_mem: float = 50.0) -> int:
        nid = len(self.neurons)
        node = NeuronNode(nid, layer_id, v_thresh, tau_mem, refractory_period=3.0, is_inhibitory=False)
        self.neurons.append(node)
        return nid

    def add_synapse(self, src_id: int, tgt_id: int, weight: float, delay: float = 1.0):
        if src_id < len(self.neurons) and tgt_id < len(self.neurons):
            syn = Synapse(tgt_id, weight, delay)
            self.neurons[src_id].outgoing_synapses.append(syn)

    def build_from_torch_model(self, model: nn.Module):
        print(f"🏗 [Kernel] Compiling model with Dale's Law (20% Inhibition)...")
        self.neurons = []
        self.spike_history = []
        
        layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear): layers.append(module)
        if not layers:
            for child in model.children():
                if isinstance(child, nn.Linear): layers.append(child)
        if not layers:
            print("❌ [Kernel] Fatal: No layers found.")
            return

        total_neurons = 0
        layer_sizes = []
        
        input_size = layers[0].in_features
        layer_sizes.append(input_size)
        for _ in range(input_size):
            self.neurons.append(NeuronNode(total_neurons, 0, v_thresh=0.2, is_inhibitory=False))
            total_neurons += 1
        
        inhibitory_ratio = 0.2
        for i, layer in enumerate(layers):
            output_size = layer.out_features
            layer_sizes.append(output_size)
            for _ in range(output_size):
                is_inhibitory = False
                if i < len(layers) - 1:
                    is_inhibitory = (random.random() < inhibitory_ratio)
                
                thresh = 0.4 if is_inhibitory else 0.5
                self.neurons.append(NeuronNode(
                    total_neurons, i + 1, 
                    v_thresh=thresh, 
                    refractory_period=3.0,
                    is_inhibitory=is_inhibitory
                ))
                total_neurons += 1
        
        print(f"   -> Created {len(self.neurons)} neurons. Inhibitory ratio ~{inhibitory_ratio*100:.0f}%")

        count_synapses = 0
        current_input_start = 0
        sparsity_threshold = 0.05 

        for i, layer in enumerate(layers):
            weights = layer.weight.detach().cpu().numpy()
            input_dim = layer.in_features
            output_dim = layer.out_features
            output_start = current_input_start + input_dim
            
            weight_scale = 5.0 
            inhibition_strength = 3.0
            
            for out_idx in range(output_dim):
                tgt_id = output_start + out_idx
                for in_idx in range(input_dim):
                    src_id = current_input_start + in_idx
                    raw_w = weights[out_idx, in_idx]
                    
                    if abs(raw_w) > sparsity_threshold:
                        src_neuron = self.neurons[src_id]
                        base_weight = abs(raw_w) * weight_scale
                        final_weight = -base_weight * inhibition_strength if src_neuron.is_inhibitory else base_weight
                        
                        delay = random.uniform(1.0, 3.0)
                        syn = Synapse(tgt_id, final_weight, delay)
                        src_neuron.outgoing_synapses.append(syn)
                        count_synapses += 1
            
            current_input_start += input_dim
            
        print(f"✅ [Kernel] Graph built: {len(self.neurons)} neurons, {count_synapses} synapses.")
        self.stats["ops"] = count_synapses

    def push_input_spikes(self, spike_indices: List[int], timestamp: float):
        for idx in spike_indices:
            if idx < len(self.neurons):
                heapq.heappush(self.event_queue, SpikeEvent(timestamp, idx, payload=5.0))
                self.spike_history.append((timestamp, idx, False))

    def run(self, duration: float = 1.0, learning_enabled: bool = True) -> Dict[int, int]:
        end_time = self.current_time + duration
        spike_counts: Dict[int, int] = {}
        
        while self.event_queue:
            if self.event_queue[0].timestamp > end_time:
                break
            
            event = heapq.heappop(self.event_queue)
            
            dt = max(0.1, event.timestamp - self.current_time)
            self.current_time = event.timestamp
            
            src_neuron = self.neurons[event.neuron_id]
            self.stats["spikes"] += 1
            spike_counts[event.neuron_id] = spike_counts.get(event.neuron_id, 0) + 1
            
            for synapse in src_neuron.outgoing_synapses:
                target_neuron = self.neurons[synapse.target_id]
                target_neuron.integrate(synapse.weight, 1.0)
                self.stats["ops"] += 1
                
                if target_neuron.check_fire(self.current_time):
                    next_time = self.current_time + synapse.delay
                    heapq.heappush(self.event_queue, SpikeEvent(next_time, target_neuron.id))
                    self.spike_history.append((self.current_time, target_neuron.id, target_neuron.is_inhibitory))
                    
                    if learning_enabled:
                        self._apply_plasticity(src_neuron, target_neuron, synapse)
        
        return spike_counts

    def _apply_plasticity(self, pre: NeuronNode, post: NeuronNode, synapse: Synapse):
        if pre.is_inhibitory: return
        if post.prediction_error > 0.1:
            synapse.weight += 0.05 * post.prediction_error
            synapse.weight = min(synapse.weight, 10.0)
            self.stats["plasticity_events"] += 1

    def reset_state(self):
        self.current_time = 0.0
        self.event_queue = []
        self.spike_history = []
        for n in self.neurons:
            n.v = 0.0
            n.last_spike_time = -100.0
            n.prediction_error = 0.0

# [Fix] Backward Compatibility Alias
EventDrivenSimulator = DORAKernel