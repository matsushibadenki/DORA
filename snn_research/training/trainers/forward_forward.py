# ファイルパス: snn_research/training/trainers/forward_forward.py
# 日本語タイトル: forward_forward
# 目的: Goodness爆発対策（Running Thresholdの導入）と入力正規化の形状修正

from snn_research.training.base_trainer import AbstractTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, List, Union, Dict, Any, Tuple
from pathlib import Path
from tqdm import tqdm
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../..')))

logger = logging.getLogger(__name__)


class SurrogateHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad = grad_output.clone() / (5.0 * torch.abs(input) + 1.0)**2
        return grad


class SpikingLayer(nn.Module):
    def __init__(self, layer: nn.Module, time_steps: int = 20, tau: float = 0.5, v_threshold: float = 1.0, reset_mechanism: str = "subtract"):
        super().__init__()
        self.layer = layer
        self.time_steps = time_steps
        self.tau = tau
        self.v_threshold = v_threshold
        self.reset_mechanism = reset_mechanism
        self.spike_fn = SurrogateHeaviside.apply

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input_dims = x.dim()
        # 入力が (B, D) の場合
        if not ((input_dims == 3) or (input_dims == 5)):
            out_static = self.layer(x)
            current = out_static.unsqueeze(1).repeat(
                1, self.time_steps, *([1] * (out_static.dim() - 1)))
        # 入力が (B, T, D) または (B, T, C, H, W) の場合
        else:
            B, T = x.shape[0], x.shape[1]
            x_flat = x.flatten(0, 1)
            out_flat = self.layer(x_flat)
            current = out_flat.view([B, T] + list(out_flat.shape[1:]))

        v_mem = torch.zeros_like(current[:, 0])
        spikes_list = []
        v_mem_list = []

        for t in range(self.time_steps):
            v_mem.mul_(self.tau).add_(current[:, t])
            spike = self.spike_fn(v_mem - self.v_threshold)

            if self.reset_mechanism == "subtract":
                v_mem.sub_(spike * self.v_threshold)
            else:
                v_mem.mul_(1.0 - spike)

            spikes_list.append(spike)
            v_mem_list.append(v_mem.clone())

        return torch.stack(spikes_list, dim=1), torch.stack(v_mem_list, dim=1)


class SpikingForwardForwardLayer(nn.Module):
    def __init__(self, forward_block, learning_rate=0.001, time_steps=20, tau=0.5, v_threshold=1.0, reset_mechanism="subtract"):
        super().__init__()
        self.block = SpikingLayer(
            forward_block,
            time_steps=time_steps,
            tau=tau,
            v_threshold=v_threshold,
            reset_mechanism=reset_mechanism
        )
        self.optimizer = torch.optim.Adam(
            self.block.parameters(), lr=learning_rate)

        # 動的閾値（初期値2.0）
        self.register_buffer("running_threshold", torch.tensor(2.0))
        self.threshold_momentum = 0.05
        
        # Conv層判定
        self.is_conv = any(isinstance(m, nn.Conv2d) for m in forward_block.modules())

    def forward(self, x):
        # 入力の正規化：SpikingLayerへの入力はノルムを1付近に保つ必要がある
        if self.is_conv:
            # (B, C, H, W) -> (B, 1, 1, 1) で正規化
            dims = [1, 2, 3] if x.dim() == 4 else [1]
            norm = x.norm(2, dim=dims, keepdim=True) + 1e-8
            x_norm = x / norm
        else:
            if x.dim() > 2:
                # (B, T, D) の場合、バッチごとに全時刻・全特徴量で正規化するか、
                # 時刻ごとに正規化するかだが、ここではバッチごとのエネルギーを一定にする
                B, T, D = x.shape
                x_flat = x.reshape(B, -1) # (B, T*D)
                norm = x_flat.norm(2, 1, keepdim=True) + 1e-8
                # 正規化後に元の形状に戻す 【重要】
                x_norm = (x_flat / norm).view(B, T, D)
            else:
                # (B, D)
                norm = x.norm(2, 1, keepdim=True) + 1e-8
                x_norm = x / norm
        
        return self.block(x_norm)

    def compute_goodness(self, hidden_spikes: torch.Tensor, hidden_v_mem: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 電位の二乗平均をGoodnessとする
        if hidden_v_mem is not None:
            return hidden_v_mem.pow(2).mean(dim=1).mean(dim=1)
        else:
            return hidden_spikes.mean(dim=1).mean(dim=1)

    def train_step(self, x_pos, x_neg):
        self.optimizer.zero_grad()

        sp_pos, v_pos = self(x_pos)
        sp_neg, v_neg = self(x_neg)

        g_pos = self.compute_goodness(sp_pos, v_pos)
        g_neg = self.compute_goodness(sp_neg, v_neg)

        # トレーニング中のみ閾値を更新
        if self.training:
            with torch.no_grad():
                mean_pos_g = g_pos.mean()
                self.running_threshold.mul_(
                    1 - self.threshold_momentum).add_(mean_pos_g * self.threshold_momentum)

        current_threshold = self.running_threshold.item()

        # Loss計算: Posは閾値より大きく、Negは閾値より小さくなるように学習
        # softplusを使って滑らかに損失を与える
        loss = F.softplus(-(g_pos - current_threshold)).mean() + \
            F.softplus(g_neg - current_threshold).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.block.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item(), g_pos.mean().item(), g_neg.mean().item()


class ForwardForwardLayer(nn.Module):
    def __init__(self, forward_block, threshold=2.0, learning_rate=0.001):
        super().__init__()
        self.block = forward_block
        self.optimizer = torch.optim.Adam(
            self.block.parameters(), lr=learning_rate)
        self.is_conv = any(isinstance(m, nn.Conv2d)
                           for m in self.block.modules())
        
        # 固定閾値ではなく、動的閾値を導入（爆発防止）
        self.register_buffer("running_threshold", torch.tensor(threshold))
        self.threshold_momentum = 0.05

    def forward(self, x):
        if self.is_conv:
            norm = x.norm(2, dim=[1, 2, 3] if x.dim() ==
                          4 else [1], keepdim=True) + 1e-8
            return self.block(x / norm)
        
        # フラット化して正規化
        x_flat = x.reshape(x.size(0), -1) if x.dim() > 2 else x
        x_norm = x_flat / (x_flat.norm(2, 1, keepdim=True) + 1e-8)
        
        # Blockを通す
        return self.block(x_norm)

    def compute_goodness(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.is_conv:
            dims = list(range(1, hidden.dim()))
            return hidden.pow(2).mean(dim=dims)
        # (B, D) -> (B,)
        return hidden.pow(2).mean(dim=1)

    def train_step(self, x_pos, x_neg):
        self.optimizer.zero_grad()
        out_pos, out_neg = self(x_pos), self(x_neg)

        g_pos = self.compute_goodness(out_pos)
        g_neg = self.compute_goodness(out_neg)

        # 動的閾値の更新
        if self.training:
            with torch.no_grad():
                mean_pos_g = g_pos.mean()
                # 異常値除外のためのガード
                if not torch.isnan(mean_pos_g) and not torch.isinf(mean_pos_g):
                    self.running_threshold.mul_(
                        1 - self.threshold_momentum).add_(mean_pos_g * self.threshold_momentum)
        
        current_threshold = self.running_threshold.item()

        # 損失関数
        loss = F.softplus(-(g_pos - current_threshold)).mean() + \
            F.softplus(g_neg - current_threshold).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.block.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item(), g_pos.mean().item(), g_neg.mean().item()


class ForwardForwardTrainer(AbstractTrainer):
    def __init__(self, model: nn.Sequential, device: str = "cpu", config: Optional[Dict[str, Any]] = None, num_classes: int = 10, save_dir: str = "results"):
        super().__init__(model, None, None, device, save_dir)
        self.num_classes = num_classes
        from omegaconf import OmegaConf
        self.config = OmegaConf.create(
            config) if config else OmegaConf.create()
        self.ff_layers: List[Union[SpikingForwardForwardLayer,
                                   ForwardForwardLayer]] = []
        self.execution_pipeline: List[nn.Module] = []
        self.use_snn = self.config.get('use_snn', False)
        self.time_steps = self.config.get('time_steps', 20)
        self.snn_tau = self.config.get('snn_tau', 0.5)
        self.snn_threshold = self.config.get('snn_threshold', 1.0)
        self.snn_reset = self.config.get('snn_reset', 'subtract')

        for layer in model.children():
            # LinearやConv2dのみForwardForwardレイヤー化する
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                lr = self.config.get("learning_rate", 0.001)

                ff: Union[SpikingForwardForwardLayer, ForwardForwardLayer]
                if self.use_snn:
                    ff = SpikingForwardForwardLayer(
                        nn.Sequential(layer),
                        learning_rate=lr,
                        time_steps=self.time_steps,
                        tau=self.snn_tau,
                        v_threshold=self.snn_threshold,
                        reset_mechanism=self.snn_reset
                    )
                else:
                    threshold = self.config.get("ff_threshold", 2.0)
                    ff = ForwardForwardLayer(nn.Sequential(
                        layer), threshold=threshold, learning_rate=lr)

                # 生成した層を明示的にデバイスへ転送
                ff = ff.to(device)

                self.ff_layers.append(ff)
                self.execution_pipeline.append(ff)
            else:
                # ReLUなどはそのままパイプラインに追加
                self.execution_pipeline.append(layer.to(device))

    def overlay_y_on_x(self, x, y):
        x_mod = x.clone()
        y_oh = F.one_hot(y, self.num_classes).float().to(self.device)
        scale_factor = 2.5

        if x_mod.dim() == 4:
            x_mod[:, 0, 0, :min(self.num_classes, x_mod.shape[3])] = y_oh[:, :min(
                self.num_classes, x_mod.shape[3])] * scale_factor
        elif x_mod.dim() == 2:
            x_mod[:, :self.num_classes] = y_oh * scale_factor
        return x_mod

    def train_epoch(self, train_loader: DataLoader, epoch: Optional[int] = None) -> Dict[str, float]:
        if epoch:
            self.current_epoch = epoch
        total_loss = 0.0
        total_pos_goodness = 0.0
        total_neg_goodness = 0.0
        num_batches = 0

        self.model.train()

        for data, target in tqdm(train_loader, desc=f"Epoch {epoch}"):
            data, target = data.to(self.device), target.to(self.device)

            x_pos = self.overlay_y_on_x(data, target)

            # Hard Negative Mining
            x_neg_base = data.roll(shifts=1, dims=0)
            x_neg_base = x_neg_base + 0.1 * torch.randn_like(x_neg_base)
            x_neg = self.overlay_y_on_x(
                x_neg_base, (target + 1) % self.num_classes)

            batch_loss = 0.0
            batch_pos_g = 0.0
            batch_neg_g = 0.0

            for layer in self.execution_pipeline:
                if isinstance(layer, (ForwardForwardLayer, SpikingForwardForwardLayer)):
                    layer_loss, g_pos, g_neg = layer.train_step(x_pos, x_neg)
                    batch_loss += layer_loss
                    batch_pos_g += g_pos
                    batch_neg_g += g_neg

                    # 次のレイヤーへの入力を作成
                    with torch.no_grad():
                        out_pos = layer(x_pos)
                        out_neg = layer(x_neg)

                        if isinstance(out_pos, tuple):
                            # SNNの場合は(spikes, v_mem)のタプルが返るためspikesを使う
                            x_pos = out_pos[0].detach()
                            x_neg = out_neg[0].detach()
                        else:
                            x_pos = out_pos.detach()
                            x_neg = out_neg.detach()
                else:
                    # ReLUなどの通常レイヤー
                    x_pos = layer(x_pos)
                    x_neg = layer(x_neg)

            total_loss += batch_loss
            total_pos_goodness += (batch_pos_g / max(1, len(self.ff_layers)))
            total_neg_goodness += (batch_neg_g / max(1, len(self.ff_layers)))
            num_batches += 1

        return {
            "train_loss": total_loss / max(1, num_batches),
            "avg_pos_goodness": total_pos_goodness / max(1, num_batches),
            "avg_neg_goodness": total_neg_goodness / max(1, num_batches)
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        acc = self.predict(val_loader)
        return {"val_accuracy": acc}

    def predict(self, test_loader: DataLoader) -> float:
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Predicting", leave=False):
                data, target = data.to(self.device), target.to(self.device)
                batch_size = data.size(0)

                data_expanded = data.repeat_interleave(self.num_classes, dim=0)
                labels_expanded = torch.arange(
                    self.num_classes, device=self.device).repeat(batch_size)

                x = self.overlay_y_on_x(data_expanded, labels_expanded)

                total_goodness = torch.zeros(
                    batch_size * self.num_classes, device=self.device)

                current_x = x
                for layer in self.execution_pipeline:
                    if isinstance(layer, (ForwardForwardLayer, SpikingForwardForwardLayer)):
                        out = layer(current_x)

                        if isinstance(layer, SpikingForwardForwardLayer):
                            spikes, v_mem = out
                            g = layer.compute_goodness(spikes, v_mem)
                            current_x = spikes.detach()
                        else:
                            g = layer.compute_goodness(out)
                            current_x = out.detach()

                        total_goodness += g
                    else:
                        current_x = layer(current_x)

                goodness_scores = total_goodness.view(
                    batch_size, self.num_classes)
                preds = goodness_scores.argmax(dim=1)

                correct += int(preds.eq(target).sum().item())
                total += batch_size

        accuracy = 100.0 * correct / total
        return accuracy

    def save_checkpoint(self, filename: str = "checkpoint.pth", metric: Optional[float] = None) -> None:
        path = Path(filename)
        parent = path.parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

        state = {
            'epoch': self.current_epoch,
            'model_state': self.model.state_dict(),
            'ff_optimizers': [layer.optimizer.state_dict() for layer in self.ff_layers],
            'best_metric': self.best_metric if metric is None else metric
        }
        torch.save(state, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            logger.warning(f"Checkpoint not found at {path}")
            return

        state = torch.load(path, map_location=self.device)
        self.current_epoch = state.get('epoch', 0)
        self.model.load_state_dict(state['model_state'])
        if 'ff_optimizers' in state:
            for layer, s in zip(self.ff_layers, state['ff_optimizers']):
                layer.optimizer.load_state_dict(s)
        logger.info(f"Loaded checkpoint from {path}")