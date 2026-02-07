# snn_research/cognitive_architecture/visual_cortex.py
# Title: Visual Cortex (Cryogenic Reset)
# Description: 
#   INFERNO後の再燃を防ぐため、リセット信号を強化。
#   強度: -10.0 -> -20.0
#   期間: 20 cycles -> 30 cycles
#   これにより、パニック後の過敏状態を完全に鎮静化させる。

import torch
import logging
from PIL import Image
from sentence_transformers import SentenceTransformer, util

class VisualCortex:
    def __init__(self, brain, model_name='clip-ViT-B-32'):
        self.brain = brain
        self.logger = logging.getLogger("VisualCortex")
        self.device = brain.device
        
        # Vision-Language Model
        self.model = SentenceTransformer(model_name)
        self.logger.info(f"👁️ Visual Cortex initialized. Model: {model_name}")
        
        # [COMPETITIVE ANCHORS]
        self.danger_concepts = ["fire", "flame", "blood", "red danger", "explosion"]
        self.safe_concepts   = ["sky", "water", "grass", "blue peaceful", "green nature"]
        
        # Anchors (CPU)
        self.danger_anchor = self.model.encode(self.danger_concepts, convert_to_tensor=True, device='cpu').mean(dim=0)
        self.safe_anchor   = self.model.encode(self.safe_concepts,   convert_to_tensor=True, device='cpu').mean(dim=0)
        
        # Gains
        self.base_gain = 0.05
        self.panic_gain = 0.5
        
        if hasattr(brain, 'input_dim'):
            self.input_dim = brain.input_dim
        else:
            self.input_dim = 128

    def process_image(self, image_input):
        # Load Image
        if isinstance(image_input, str):
            try:
                img = Image.open(image_input)
            except Exception as e:
                self.logger.error(f"Failed to load image: {e}")
                return []
        else:
            img = image_input

        # Encode (CPU)
        embedding = self.model.encode(img, convert_to_tensor=True, device='cpu')
        
        # [COMPETITIVE CHECK]
        danger_score = util.cos_sim(embedding, self.danger_anchor).item()
        safe_score   = util.cos_sim(embedding, self.safe_anchor).item()
        
        # 危険判定 (マージン +0.02)
        is_danger = danger_score > (safe_score + 0.02)
        current_gain = self.panic_gain if is_danger else self.base_gain
        
        status = "🚨 DANGER" if is_danger else "ℹ️ SAFE"
        print(f"👁️ [VisualCortex] Score: Danger={danger_score:.3f} vs Safe={safe_score:.3f} -> [{status}]")

        # Brain Input Construction
        if embedding.shape[0] > self.input_dim:
            signal = embedding[:self.input_dim]
        else:
            padding = torch.zeros(self.input_dim - embedding.shape[0], device='cpu')
            signal = torch.cat([embedding, padding])
            
        signal = signal.to(self.device).unsqueeze(0)
        
        # Rectification
        signal = torch.nn.functional.normalize(signal, p=2, dim=1)
        signal = torch.abs(signal) * current_gain

        # --- Protocol Execution ---
        self._apply_deep_freeze() # 強力リセット実行
        
        spike_counts = []
        for cycle in range(20):
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
        print(f"   -> Visual Output: Avg Spikes={avg:.2f}")
        return spike_counts

    def _apply_deep_freeze(self):
        """
        Cryogenic Reset:
        強度 -20.0 (絶対零度級) で 30サイクル冷却し、
        さらに 20サイクルの安定化(Zero Input)を設ける。
        """
        # Phase 1: Deep Inhibition
        inhibitory = torch.ones(1, self.input_dim, device=self.device) * -20.0
        for _ in range(30):
            self.brain.process_step(inhibitory)
            
        # Phase 2: Long Settle
        zero = torch.zeros(1, self.input_dim, device=self.device)
        for _ in range(20):
            self.brain.process_step(zero)