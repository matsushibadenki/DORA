# directory: snn_research/training/trainers
# file: physics_informed.py
# title: Physics Informed Trainer
# description: 物理法則の整合性制約を損失関数に組み込んだトレーナーの実装。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union

from snn_research.training.base_trainer import BaseTrainer
from snn_research.cognitive_architecture.physics_evaluator import PhysicsEvaluator

class PhysicsInformedTrainer(BaseTrainer):
    """
    物理法則制約付きトレーナー。
    PhysicsEvaluatorを使用して、モデルの内部状態や出力が物理法則（エネルギー保存則や運動法則など）
    と整合しているかを評価し、その逸脱を損失として加算します。
    """

    def __init__(self, 
                 model: nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 config: Dict[str, Any], 
                 device: torch.device = None):
        """
        Args:
            model: 学習対象のモデル
            optimizer: オプティマイザ
            config: 設定辞書。'physics'キー配下にPhysicsEvaluatorの設定を含めることを推奨。
            device: 実行デバイス
        """
        super().__init__(model, optimizer, config, device)
        
        # 物理評価エンジンの初期化
        physics_config = config.get("physics", {})
        self.physics_evaluator = PhysicsEvaluator(physics_config)
        
        # 物理損失の重み
        self.physics_loss_weight = config.get("physics_loss_weight", 0.1)

    def compute_physics_loss(self, states: torch.Tensor) -> torch.Tensor:
        """
        モデルの内部状態から物理整合性損失を計算します。
        
        Args:
            states: モデルの内部状態シーケンス (Batch, Time, Features)
            
        Returns:
            loss: スカラ損失値
        """
        # PhysicsEvaluatorを用いて整合性を計算
        # evaluate_state_consistencyは整合性スコア（高いほど良い）を返すと仮定し、
        # 損失にするために反転または負の対数をとるなどの処理を行います。
        # 現状のPhysicsEvaluatorが勾配計算に対応していない場合は、
        # ここで独自の微分可能なペナルティ項を実装する必要があります。
        
        # 簡易的な実装: 状態の急激な変化（非物理的な動き）に対する正則化
        if states.shape[1] > 1:
            diff = states[:, 1:] - states[:, :-1]
            smoothness_loss = torch.mean(diff ** 2)
            return smoothness_loss
        
        return torch.tensor(0.0, device=self.device)

    def train_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        """
        1ステップの学習を実行します。タスク損失に物理損失を加えます。
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # モデルの順伝播
        # モデルによっては (output, states) を返す場合や dict を返す場合があるため分岐
        outputs = self.model(inputs)
        
        states = None
        task_loss = 0.0
        
        if isinstance(outputs, dict):
            logits = outputs.get("logits", outputs.get("output"))
            states = outputs.get("states", outputs.get("state_history"))
            if "loss" in outputs:
                task_loss = outputs["loss"]
            elif logits is not None:
                task_loss = nn.functional.cross_entropy(logits, targets)
        else:
            logits = outputs
            task_loss = nn.functional.cross_entropy(logits, targets)
            
            # モデルがメソッドで状態を公開している場合
            if hasattr(self.model, "get_state_history"):
                states = self.model.get_state_history()

        # 物理損失の計算と加算
        total_loss = task_loss
        physics_loss_value = 0.0
        
        if states is not None and self.physics_loss_weight > 0:
            physics_loss = self.compute_physics_loss(states)
            total_loss += self.physics_loss_weight * physics_loss
            physics_loss_value = physics_loss.item()

        # 逆伝播と更新
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "loss": total_loss.item(),
            "task_loss": task_loss.item() if isinstance(task_loss, torch.Tensor) else task_loss,
            "physics_loss": physics_loss_value
        }