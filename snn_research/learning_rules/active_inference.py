# directory: snn_research/learning_rules
# file: active_inference.py
# title: Active Inference Rule (SARA Engine Integrated)
# purpose: 能動的推論（Active Inference）アルゴリズムの実装。
#          従来の単独計算ロジックに加え、SARA Engine v7.4の高度な内部ループへの委譲をサポート。

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Any, Union, Tuple

# SARA Engineのインポート試行（プロジェクト構造に依存）
try:
    from snn_research.models.experimental.sara_engine import SARAEngine
except ImportError:
    SARAEngine = None

class ActiveInferenceRule:
    """
    能動的推論（Active Inference）を実行するクラス。
    
    環境からの感覚入力（Observation）を受け取り、自由エネルギー（Free Energy）を最小化するように
    内部状態（Belief）の更新と、外界への作用（Action）の生成を行う。
    
    Modes:
    1. SARA Backend Mode: 対象モデルがSARA Engineの場合、計算をエンジンに委譲する。
    2. Standalone Mode: 通常のPyTorchモデルに対し、勾配降下法を用いて推論を行う（Legacy）。
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        action_rate: float = 1e-2,
        precision_rate: float = 0.5,
        use_sara_backend: bool = True,
        **kwargs
    ):
        """
        Args:
            model (nn.Module): 推論を行うモデル
            learning_rate (float): 知覚（Weights/Belief）の更新率
            action_rate (float): 行動（Action）の更新率
            precision_rate (float): 予測誤差の精度重み（Precision）
            use_sara_backend (bool): SARAエンジンが検出された場合に委譲するかどうか
        """
        self.model = model
        self.lr = learning_rate
        self.action_lr = action_rate
        self.precision_rate = precision_rate
        
        # SARA Backendの検出
        self.sara_backend = None
        if use_sara_backend:
            if SARAEngine and isinstance(model, SARAEngine):
                self.sara_backend = model
            elif hasattr(model, 'engine') and SARAEngine and isinstance(model.engine, SARAEngine):
                # PredictiveCodingModelなどでラップされている場合
                self.sara_backend = model.engine
        
        if self.sara_backend:
            print(f"ActiveInferenceRule: Connected to SARA Engine v{getattr(self.sara_backend, 'version', 'Unknown')}")
            # SARAを能動的推論モードに設定
            if hasattr(self.sara_backend, 'set_perception_mode'):
                self.sara_backend.set_perception_mode('active_inference')
        else:
            print("ActiveInferenceRule: Running in Standalone (Legacy) Mode.")
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            self.loss_fn = nn.MSELoss(reduction='none')

    def step(
        self, 
        observation: torch.Tensor, 
        goal: Optional[torch.Tensor] = None,
        return_details: bool = False
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        1タイムステップの推論と行動生成を実行する。

        Args:
            observation (torch.Tensor): 現在の観測 [Batch, Input_Dim]
            goal (torch.Tensor, optional): 目標とする観測状態（Preferred Observation）。
                                           Noneの場合は単なる予測誤差最小化（受動的知覚）となる。
            return_details (bool): 詳細なステータスを辞書で返すかどうか

        Returns:
            action (torch.Tensor) or details (Dict)
        """
        
        if self.sara_backend:
            return self._step_sara(observation, goal, return_details)
        else:
            return self._step_legacy(observation, goal, return_details)

    def _step_sara(self, observation: torch.Tensor, goal: Optional[torch.Tensor], return_details: bool):
        """SARA Engineを使用した処理"""
        # SARAの内部メソッドを呼び出して推論ステップを進める
        # 想定されるインターフェース: process_step(input, goal, mode) -> dict
        
        results = self.sara_backend.process_step(
            input_data=observation,
            target_goal=goal,
            require_action=True
        )
        
        action = results.get("action", torch.zeros_like(observation))
        
        if return_details:
            return {
                "action": action,
                "prediction": results.get("prediction"),
                "free_energy": results.get("free_energy", 0.0),
                "belief_state": results.get("hidden_state")
            }
        return action

    def _step_legacy(self, observation: torch.Tensor, goal: Optional[torch.Tensor], return_details: bool):
        """
        従来の勾配ベースの実装（SARAがない場合用）。
        入力に対する勾配を計算してアクションとする簡易的な実装。
        """
        self.model.eval() # 勾配計算のためEvalモードだが、入力の勾配はとる
        
        # アクション（あるいは入力の修正量）を最適化対象とする
        # 実際にはロボットアームのトルクなどがActionだが、ここでは単純化して
        # 「入力をどう変えたいか」をActionとみなす実装例
        observation_var = observation.clone().detach().requires_grad_(True)
        
        # 1. 知覚: 現在の状態からの予測
        prediction = self.model(observation_var)
        
        # 2. 評価: 目標がある場合は目標との誤差、なければ予測誤差自体を最小化（受動的）
        target = goal if goal is not None else observation
        
        # 自由エネルギー（変分自由エネルギーの近似としての予測誤差）
        # F ≈ 1/2 * Precision * (Observation - Prediction)^2
        prediction_error = target - prediction
        free_energy = 0.5 * self.precision_rate * torch.sum(prediction_error ** 2)
        
        # 3. 行動生成: 自由エネルギーを入力(Action)について微分
        # Action = - k * dF/dm (m=motor command, here approximated by observation input change)
        free_energy.backward()
        action_grad = observation_var.grad
        
        # 勾配降下法によるアクション生成
        action = -self.action_lr * action_grad
        
        # 4. 学習: 重みの更新（VFE最小化）
        if self.model.training:
            self.optimizer.zero_grad()
            # 再計算（グラフがつながっている必要があるため）
            pred_train = self.model(observation)
            loss = 0.5 * self.loss_fn(pred_train, target).sum()
            loss.backward()
            self.optimizer.step()
        
        if return_details:
            return {
                "action": action.detach(),
                "prediction": prediction.detach(),
                "free_energy": free_energy.item(),
                "error": prediction_error.detach()
            }
        return action.detach()

    def reset(self):
        """内部状態のリセット"""
        if self.sara_backend:
            self.sara_backend.reset()