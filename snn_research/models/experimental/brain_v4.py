# ファイルパス: snn_research/models/experimental/brain_v4.py
# 日本語タイトル: Brain v4.1 Synesthetic Architecture (Generate Fix)
# 修正内容: generateメソッド内で return_spikes=True を指定し、アンパックエラーを修正。

import torch
import torch.nn as nn
from typing import Optional, Dict

from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
from snn_research.io.universal_encoder import UniversalSpikeEncoder
from snn_research.hybrid.multimodal_projector import UnifiedSensoryProjector


class SynestheticBrain(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 6,
        time_steps: int = 4,
        tactile_dim: int = 64,
        olfactory_dim: int = 32,
        device: str = 'cpu'
    ):
        super().__init__()
        self.device = device
        self.time_steps = time_steps
        self.d_model = d_model

        # 1. 汎用感覚野
        self.encoder = UniversalSpikeEncoder(
            time_steps=time_steps,
            output_dim=d_model,
            device=device
        ).to(device)

        # 2. 五感統合ブリッジ
        modality_configs = {
            'vision': d_model,
            'audio': d_model,
            'tactile': d_model,
            'olfactory': d_model
        }

        self.sensory_bridge = UnifiedSensoryProjector(
            language_dim=d_model,
            modality_configs=modality_configs,
            use_bitnet=False
        ).to(device)

        # 3. 前頭葉・中枢エンジン
        self.core_brain = BitSpikeMamba(
            vocab_size=vocab_size,
            d_model=d_model,
            d_state=16, d_conv=4, expand=2,
            num_layers=num_layers,
            time_steps=time_steps,
            neuron_config={"type": "lif", "tau_mem": 2.0}
        ).to(device)

    def forward(self,
                text_input: Optional[torch.Tensor] = None,
                image_input: Optional[torch.Tensor] = None,
                audio_input: Optional[torch.Tensor] = None,
                tactile_input: Optional[torch.Tensor] = None,
                olfactory_input: Optional[torch.Tensor] = None):
        """
        五感入力をスパイク化し、統合ブリッジを通してコンテキスト化する。
        """
        # Phase 1: 各感覚器でのスパイク符号化
        sensory_spikes: Dict[str, torch.Tensor] = {}

        if image_input is not None:
            sensory_spikes['vision'] = self.encoder.encode(
                image_input, modality='image')

        if audio_input is not None:
            sensory_spikes['audio'] = self.encoder.encode(
                audio_input, modality='audio')

        if tactile_input is not None:
            sensory_spikes['tactile'] = self.encoder.encode(
                tactile_input, modality='tactile')

        if olfactory_input is not None:
            sensory_spikes['olfactory'] = self.encoder.encode(
                olfactory_input, modality='olfactory')

        # Phase 2: 感覚統合
        sensory_context = self.sensory_bridge(sensory_spikes)

        # Phase 3: 言語と思考
        if text_input is not None:
            text_emb = self.core_brain.embedding(text_input)
            combined_input = torch.cat([sensory_context, text_emb], dim=1)
        else:
            if sensory_context.size(1) == 0:
                raise ValueError("No input provided to SynestheticBrain")
            combined_input = sensory_context

        # Phase 4: 統合処理
        logits, _, _ = self.core_brain(combined_input, return_spikes=True)

        return logits

    def generate(self, image_input: torch.Tensor, start_token_id: int, max_new_tokens: int = 20):
        self.eval()
        with torch.no_grad():
            # 視覚を知覚 -> 統合コンテキストへ
            vision_spikes = self.encoder.encode(image_input, modality='image')
            current_context = self.sensory_bridge({'vision': vision_spikes})

            generated_ids = []
            curr_input_ids = torch.tensor(
                [[start_token_id]], device=self.device)

            for _ in range(max_new_tokens):
                text_emb = self.core_brain.embedding(curr_input_ids)
                combined = torch.cat([current_context, text_emb], dim=1)
                
                # [Fix] return_spikes=True を追加して戻り値の数を合わせる
                logits, _, _ = self.core_brain(combined, return_spikes=True)

                next_token = torch.argmax(
                    logits[:, -1, :], dim=-1, keepdim=True)
                generated_ids.append(next_token.item())
                curr_input_ids = torch.cat([curr_input_ids, next_token], dim=1)

        return generated_ids