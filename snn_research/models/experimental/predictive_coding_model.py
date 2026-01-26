# ファイルパス: snn_research/models/experimental/predictive_coding_model.py
import torch
import torch.nn as nn
from typing import List, Dict, Any, cast, Optional, Tuple, Union

from snn_research.core.networks.sequential_pc_network import SequentialPCNetwork
from snn_research.core.layers.predictive_coding import PredictiveCodingLayer
from snn_research.core.neurons import AdaptiveLIFNeuron
from snn_research.core.learning_rules.predictive_coding_rule import PredictiveCodingRule
from spikingjelly.activation_based import functional as SJ_F # type: ignore

class PredictiveCodingModel(nn.Module):
    """
    画像分類・テキスト処理兼用の予測符号化SNNモデル。
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int], 
        output_dim: int, 
        neuron_params: Dict[str, Any],
        vocab_size: Optional[int] = None
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = hidden_dims[0] if hidden_dims else input_dim
        
        # --- 入力層 ---
        self.token_embedding: Optional[nn.Embedding]
        if vocab_size is not None:
            self.token_embedding = nn.Embedding(vocab_size, self.d_model)
        else:
            self.token_embedding = None
            
        self.input_projector = nn.Linear(input_dim, self.d_model)
        
        # --- PC Network ---
        layers: List[PredictiveCodingLayer] = []
        current_dim = self.d_model
        
        for h_dim in hidden_dims:
            # 修正: d_model/d_state -> input_size/hidden_size
            layer = PredictiveCodingLayer(
                input_size=current_dim, # Bottom-up input dimension
                hidden_size=h_dim,      # Top-down state dimension
                neuron_class=AdaptiveLIFNeuron,
                neuron_params=neuron_params,
                weight_tying=True,
                learning=True # 学習ルールを使用するため
            )
            layers.append(layer)
            current_dim = h_dim
            
        self.network = SequentialPCNetwork(layers)
        self.classifier = nn.Linear(current_dim, output_dim)
        
        self._register_learning_rules()

    def _register_learning_rules(self) -> None:
        if hasattr(self.network, 'pc_layers'):
            for i, layer_module in enumerate(self.network.pc_layers):
                layer_name = f"layer_{i}"
                layer = cast(PredictiveCodingLayer, layer_module)
                
                params: List[nn.Parameter] = [cast(nn.Parameter, layer.generative_fc.weight)]
                if layer.generative_fc.bias is not None:
                    params.append(cast(nn.Parameter, layer.generative_fc.bias))
                
                rule = PredictiveCodingRule(
                    params=params,
                    learning_rate=0.005,
                    layer_name=layer_name,
                    error_weight=1.0,
                    weight_decay=1e-4
                )
                self.network.add_learning_rule(rule)

    def forward(
        self, 
        x: Optional[torch.Tensor] = None, 
        input_ids: Optional[torch.Tensor] = None, 
        labels: Optional[torch.Tensor] = None,
        return_spikes: bool = False,
        **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        inputs: torch.Tensor
        original_shape: Optional[Tuple[int, int]] = None
        
        target_ids = input_ids
        target_x = x
        
        if target_x is not None and not target_x.is_floating_point():
            if target_ids is None:
                target_ids = target_x
                target_x = None
        
        if target_ids is not None:
            if self.token_embedding is None:
                raise ValueError("Model initialized without vocab_size.")
            inputs = self.token_embedding(target_ids)
            if inputs.ndim == 3:
                b, s, d = inputs.shape
                original_shape = (b, s)
                inputs = inputs.view(b * s, d)
                
        elif target_x is not None:
            if target_x.ndim > 2:
                target_x = target_x.view(target_x.size(0), -1)
            if not target_x.is_floating_point():
                target_x = target_x.float()
            inputs = self.input_projector(target_x)
        else:
            raise ValueError("Either x or input_ids must be provided.")

        final_state = self.network(inputs)
        logits = self.classifier(final_state)
        
        if original_shape is not None:
            b, s = original_shape
            logits = logits.view(b, s, -1)
        
        if return_spikes:
            if original_shape is not None:
                b, s = original_shape
                dummy_spikes = torch.zeros(b, s, self.output_dim, device=logits.device)
            else:
                batch_size = logits.size(0)
                dummy_spikes = torch.zeros(batch_size, self.output_dim, device=logits.device)
            return logits, dummy_spikes
            
        return logits

    def reset_state(self) -> None:
        if hasattr(self.network, 'reset_state'):
            self.network.reset_state()
        SJ_F.reset_net(self)
        self.reset_spike_stats()

    def get_total_spikes(self) -> float:
        total_spikes = 0.0
        if hasattr(self.network, 'pc_layers'):
            for layer in self.network.pc_layers:
                # 属性チェックとキャスト
                gen_neuron = getattr(layer, 'generative_neuron', None)
                if gen_neuron and hasattr(gen_neuron, 'total_spikes'):
                    gen_spikes = cast(torch.Tensor, gen_neuron.total_spikes)
                    total_spikes += float(gen_spikes.item())
                    
                inf_neuron = getattr(layer, 'inference_neuron', None)
                if inf_neuron and hasattr(inf_neuron, 'total_spikes'):
                    inf_spikes = cast(torch.Tensor, inf_neuron.total_spikes)
                    total_spikes += float(inf_spikes.item())
        return total_spikes

    def reset_spike_stats(self) -> None:
        if hasattr(self.network, 'pc_layers'):
            for layer in self.network.pc_layers:
                gen_neuron = getattr(layer, 'generative_neuron', None)
                if gen_neuron and hasattr(gen_neuron, 'total_spikes'):
                    cast(torch.Tensor, gen_neuron.total_spikes).zero_()
                    
                inf_neuron = getattr(layer, 'inference_neuron', None)
                if inf_neuron and hasattr(inf_neuron, 'total_spikes'):
                    cast(torch.Tensor, inf_neuron.total_spikes).zero_()