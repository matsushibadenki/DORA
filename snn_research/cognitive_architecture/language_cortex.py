# snn_research/cognitive_architecture/language_cortex.py
# Title: Language Cortex (Deep Freeze Protocol)
# Description: 
#   パニック後の残留興奮(Panic Hangover)を防ぐため、リセット信号を
#   -2.0から-10.0へと劇的に強化。
#   これにより、連続した入力でも前の感情を引きずらず、
#   常にクリアな状態で判断できるようにする。

import torch
import logging
from sentence_transformers import SentenceTransformer

class LanguageCortex:
    def __init__(self, brain, model_name='all-MiniLM-L6-v2'):
        self.brain = brain
        self.logger = logging.getLogger("LanguageCortex")
        self.device = brain.device
        self.embedder = SentenceTransformer(model_name)
        
        # Stability Protocol (Disabled)
        self.bias_impulse = 0.0
        self.bias_sustain = 0.0
        self.impulse_duration = 0 
        self.process_duration = 20
        
        # [NANO-CURRENT GAIN]
        self.emergency_keywords = ["FIRE", "DANGER", "ALERT", "STOP", "RUN"]
        self.base_gain = 0.05   # Target: 0.00 (Silence)
        self.panic_gain = 0.5   # Target: >15.00 (Action)

        print(f"🗣️ [LanguageCortex] Initialized. Gains: {self.base_gain}/{self.panic_gain}")

        if hasattr(brain, 'input_dim'):
            self.input_dim = brain.input_dim
        elif hasattr(brain, 'topology') and hasattr(brain.topology, 'input_dim'):
            self.input_dim = brain.topology.input_dim
        else:
            self.input_dim = 128

    def process_text(self, text: str):
        # 1. Apply Deep Freeze (Reset) FIRST
        # 前回の興奮を完全に消すために、入力処理の「最初」に強力な冷却を行う
        self._apply_deep_freeze()

        is_emergency = any(kw in text.upper() for kw in self.emergency_keywords)
        current_gain = self.panic_gain if is_emergency else self.base_gain
        
        status = "🚨 PANIC" if is_emergency else "ℹ️ Normal"
        print(f"📥 [LanguageCortex] Input: '{text}' [{status}] Gain={current_gain}")

        embedding = self.embedder.encode(text, convert_to_tensor=True, device='cpu') 
        
        if embedding.shape[0] > self.input_dim:
            signal = embedding[:self.input_dim]
        else:
            padding = torch.zeros(self.input_dim - embedding.shape[0], device='cpu')
            signal = torch.cat([embedding, padding])
            
        signal = signal.to(self.device).unsqueeze(0)
        
        # Rectification
        signal = torch.nn.functional.normalize(signal, p=2, dim=1)
        signal = torch.abs(signal) * current_gain

        spike_counts = []
        for cycle in range(self.process_duration):
            # Input Construction
            input_current = signal + (torch.randn_like(signal) * 0.01)
            
            output = self.brain.process_step(input_current)
            
            count = 0
            if isinstance(output, dict) and 'output' in output:
                if isinstance(output['output'], torch.Tensor):
                    count = output['output'].sum().item()
            elif isinstance(output, torch.Tensor):
                count = output.sum().item()
                
            spike_counts.append(count)
        
        avg = sum(spike_counts) / len(spike_counts) if spike_counts else 0
        print(f"   -> Cortex Output: Avg Spikes={avg:.2f}")
        return spike_counts

    def _apply_deep_freeze(self):
        """
        Deep Freeze Protocol:
        強力な抑制電流(-10.0)を長期間(20 cycles)流し、
        残留電位を完全に消去する。
        """
        # print("   ❄️ Applying Deep Freeze Reset...")
        inhibitory = torch.ones(1, self.input_dim, device=self.device) * -10.0
        for _ in range(20):
            self.brain.process_step(inhibitory)
            
        # Settle (Zero input) to return to resting potential
        zero = torch.zeros(1, self.input_dim, device=self.device)
        for _ in range(10):
            self.brain.process_step(zero)