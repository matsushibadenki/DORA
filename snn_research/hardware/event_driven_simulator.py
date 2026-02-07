# snn_research/hardware/event_driven_simulator.py
# Title: DORA Kernel v3.8 (Input Ready)
# Description: 
#   push_input_spikes メソッドを追加し、外部からのデータ注入を可能にする。
#   AttributeError: 'DORAKernel' object has no attribute 'push_input_spikes' を解消。

import heapq
import logging
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Set
import torch
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["DORAKernel", "EventDrivenSimulator", "SpikeEvent", "NeuronNode", "Synapse"]

@dataclass(order=True)
class SpikeEvent:
    timestamp: float
    neuron_id: int = field(compare=False)
    payload: float = field(compare=False, default=1.0)

class Synapse:
    __slots__ = ('target_id', 'weight', 'delay', 'is_plastic', 'age', 'is_tagged')
    def __init__(self, target_id: int, weight: float, delay: float = 1.0, is_plastic: bool = True):
        self.target_id = target_id; self.weight = weight; self.delay = delay
        self.is_plastic = is_plastic; self.age = 0; self.is_tagged = False

class NeuronNode:
    __slots__ = ('id', 'layer_id', 'v', 'v_thresh', 'v_reset', 'v_rest', 'tau_mem', 'last_spike_time', 'outgoing_synapses', 'refractory_period', 'is_inhibitory', 'activity_trace', 'homeostatic_bias', 'fatigue_level')
    def __init__(self, neuron_id, layer_id, v_thresh=0.5, tau_mem=20.0, refractory_period=3.0, is_inhibitory=False):
        self.id = neuron_id; self.layer_id = layer_id; self.v = 0.0; self.v_rest = 0.0; self.v_thresh = v_thresh; self.v_reset = 0.0; self.tau_mem = tau_mem; self.last_spike_time = -100.0; self.refractory_period = refractory_period; self.is_inhibitory = is_inhibitory; self.activity_trace = 0.0; self.homeostatic_bias = 0.0; self.fatigue_level = 0.0; self.outgoing_synapses = []

    def integrate(self, weight, dt):
        decay = math.exp(-dt / self.tau_mem)
        self.v = (self.v - self.v_rest) * decay + self.v_rest + weight
        if self.v > 10.0: self.v = 10.0
        elif self.v < -5.0: self.v = -5.0

    def check_fire(self, current_time):
        eff_ref = self.refractory_period * (1.0 + self.fatigue_level * 2.0)
        if (current_time - self.last_spike_time) < eff_ref: return False
        eff_thresh = self.v_thresh + self.homeostatic_bias + (self.fatigue_level * 0.5)
        if self.v >= eff_thresh:
            self.v = self.v_reset; self.last_spike_time = current_time
            self.activity_trace += 1.0; self.fatigue_level += 0.02
            return True
        return False

    def recover_fatigue(self, amount):
        self.fatigue_level = max(0.0, self.fatigue_level - amount)
        self.activity_trace *= 0.95

    def update_homeostasis(self, target_rate=0.1):
        if self.activity_trace > target_rate: self.homeostatic_bias += 0.005
        else: self.homeostatic_bias -= 0.001
        self.homeostatic_bias = max(-0.5, min(3.0, self.homeostatic_bias))

class DORAKernel:
    def __init__(self, dt=1.0):
        self.neurons = []; self.event_queue = []; self.current_time = 0.0; self.dt = dt
        self.stats = {"ops": 0, "spikes": 0, "plasticity_events": 0, "synapses_created": 0, "synapses_pruned": 0, "current_synapses": 0, "surprise_index": 0.0}
        self.structural_plasticity_enabled = True; self.is_sleeping = False
        self.pruning_threshold = 0.1; self.growth_probability = 0.005
        logger.info("🧠 DORA Kernel v3.8 (Input Ready) initialized")

    def create_layer_neurons(self, count, layer_id, v_thresh=0.5):
        start = len(self.neurons)
        for _ in range(count):
            inhib = (random.random() < 0.2)
            self.neurons.append(NeuronNode(len(self.neurons), layer_id, v_thresh=v_thresh*(1.5 if inhib else 1.0), is_inhibitory=inhib))
        return start, len(self.neurons)

    def connect_groups(self, src_range, tgt_range, weights=None, sparsity=0.1):
        src_start, src_end = src_range; tgt_start, tgt_end = tgt_range; syn_count = 0
        if weights is not None:
            rows, cols = weights.shape
            for r in range(min(rows, tgt_end - tgt_start)):
                for c in range(min(cols, src_end - src_start)):
                    if abs(weights[r, c]) > 0.01:
                        self._add_synapse(src_start+c, tgt_start+r, weights[r, c], 5.0); syn_count += 1
        else:
            for src_id in range(src_start, src_end):
                targets = random.sample(range(tgt_start, tgt_end), max(1, int((tgt_end-tgt_start)*sparsity)))
                for t in targets: self._add_synapse(src_id, t, random.uniform(0.5, 1.5), 5.0); syn_count += 1
        return syn_count

    def _add_synapse(self, src, tgt, raw, scale):
        if src < len(self.neurons):
            n = self.neurons[src]
            n.outgoing_synapses.append(Synapse(tgt, (-abs(raw) if n.is_inhibitory else abs(raw))*scale, delay=random.uniform(1.0, 3.0)))

    def push_input_spikes(self, neuron_indices: List[int], timestamp: float):
        """[New] 外部からのスパイク入力をイベントキューに追加する"""
        for nid in neuron_indices:
            if nid < len(self.neurons):
                # 入力スパイクは即時発火イベントとして扱う
                heapq.heappush(self.event_queue, SpikeEvent(timestamp, nid))

    def run(self, duration=1.0, learning_enabled=True):
        end_time = self.current_time + duration; counts = {}; active = []
        while self.event_queue and self.event_queue[0].timestamp <= end_time:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.timestamp
            if event.neuron_id >= len(self.neurons): continue
            src = self.neurons[event.neuron_id]; self.stats["spikes"] += 1; active.append(event.neuron_id)
            counts[event.neuron_id] = counts.get(event.neuron_id, 0) + 1
            if learning_enabled:
                src.last_spike_time = self.current_time
                if not self.is_sleeping and self.structural_plasticity_enabled: self._attempt_synaptogenesis(src)
            for syn in src.outgoing_synapses:
                if syn.target_id >= len(self.neurons): continue
                tgt = self.neurons[syn.target_id]; tgt.integrate(syn.weight, 1.0); self.stats["ops"] += 1; syn.age += 1
                if tgt.check_fire(self.current_time):
                    heapq.heappush(self.event_queue, SpikeEvent(self.current_time + syn.delay, tgt.id))
                    if learning_enabled: self._apply_stdp(src, tgt, syn)
        if len(self.neurons) > 0: self.stats["surprise_index"] = len(set(active)) / len(self.neurons)
        if self.stats["ops"] > 0 and self.stats["ops"] % 20000 == 0: self._perform_maintenance(learning_enabled)
        return counts

    def _apply_stdp(self, pre, post, syn):
        if not syn.is_plastic or pre.is_inhibitory: return
        dt = post.last_spike_time - pre.last_spike_time
        if 0 < dt < 20.0:
            syn.weight += 0.005 * math.exp(-dt / 20.0)
            if syn.weight > 8.0: syn.is_tagged = True
            syn.age = 0; self.stats["plasticity_events"] += 1

    def _attempt_synaptogenesis(self, src):
        if random.random() > self.growth_probability: return
        target = self.neurons[random.randint(0, len(self.neurons)-1)]
        if target.id != src.id and (self.current_time - target.last_spike_time) < 5.0:
            src.outgoing_synapses.append(Synapse(target.id, -0.5 if src.is_inhibitory else 0.5)); self.stats["synapses_created"] += 1

    def _perform_maintenance(self, learning):
        total = 0; pruned = 0
        for n in self.neurons:
            if self.is_sleeping: n.recover_fatigue(0.2)
            else: n.recover_fatigue(0.002); n.update_homeostasis()
            if learning and self.structural_plasticity_enabled and not self.is_sleeping:
                orig = len(n.outgoing_synapses)
                n.outgoing_synapses = [s for s in n.outgoing_synapses if s.is_tagged or abs(s.weight) > self.pruning_threshold or s.age < 500]
                pruned += (orig - len(n.outgoing_synapses))
            total += len(n.outgoing_synapses)
        self.stats["synapses_pruned"] += pruned; self.stats["current_synapses"] = total

    def apply_synaptic_scaling(self, factor=0.85):
        for n in self.neurons:
            for s in n.outgoing_synapses:
                if s.is_plastic and not s.is_tagged: s.weight *= factor

    def reset_state(self):
        self.current_time = 0.0; self.event_queue = []
        for n in self.neurons: n.v = 0.0; n.last_spike_time = -100.0; n.fatigue_level = 0.0

EventDrivenSimulator = DORAKernel