# directory: snn_research/models/experimental
# file: predictive_coding_model.py
# title: 予測符号化モデル (SARA Engine Backend)
# purpose: 従来の予測符号化モデルのインターフェースを維持しつつ、内部ロジックを次世代のSARA Engine v7.4に差し替えたラッパーモデル。
#          高度な構造可塑性と再帰的推論能力を活用して予測精度を向上させる。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

# SARA EngineとAdapterのインポート
# ※ プロジェクト構造に基づいてパスを指定しています
try:
    from snn_research.models.experimental.sara_engine import SARAEngine
    from snn_research.models.adapters.sara_adapter import SARAAdapter
except ImportError:
    # 開発環境等でSARAが見つからない場合のフォールバック（またはエラー表示）
    print("Warning: SARA Engine not found. Please ensure snn_research.models.experimental.sara_engine exists.")
    SARAEngine = None
    SARAAdapter = None

class PredictiveCodingModel(nn.Module):
    """
    SARA Engine v7.4 をバックエンドに使用した予測符号化モデル。
    旧来の PredictiveCodingModel との互換性を維持するためのラッパーとして機能します。
    """

    def __init__(
        self, 
        input_size: int, 
        hidden_size: int = 128, 
        output_size: Optional[int] = None, 
        layer_sizes: Optional[list] = None,
        use_sara_backend: bool = True,
        **kwargs
    ):
        """
        Args:
            input_size (int): 入力次元数
            hidden_size (int): 隠れ層のサイズ (SARAの内部状態サイズ)
            output_size (int, optional): 出力次元数 (指定がない場合はinput_sizeと同じ)
            layer_sizes (list, optional): 互換性のために残している引数 (SARAでは自動構成されるため無視またはヒントとして使用)
            use_sara_backend (bool): SARAエンジンを使用するかどうかのフラグ
            **kwargs: SARA Engineに渡すその他の設定パラメータ
        """
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size if output_size is not None else input_size
        self.hidden_size = hidden_size
        
        if use_sara_backend and SARAEngine is not None:
            print(f"Initializing PredictiveCodingModel with SARA Engine Backend (Input: {input_size}, Hidden: {hidden_size})")
            
            # SARA Engineの初期化
            # 予測符号化タスクに特化した設定でインスタンス化
            self.engine = SARAEngine(
                input_dim=self.input_size,
                hidden_dim=self.hidden_size,
                version="v7.4",  # 推奨バージョンを指定
                enable_recursion=True,  # 再帰処理を有効化（精度向上）
                perception_mode="predictive",  # 知覚モードを予測型に設定
                **kwargs
            )
            
            # Adapterを使用して、SARAの複雑な出力を単純なTensor出力に変換する
            # これにより、既存のtrainループなどがそのまま動作する
            self.adapter = SARAAdapter(
                target_engine=self.engine,
                output_mode="prediction_error"  # 予測誤差または再構成画像を出力するように設定
            )
            
            self.backend_type = "sara"
            
        else:
            # フォールバック: SARAが使えない場合の簡易的な予測符号化実装 (Legacy)
            print("Warning: Falling back to legacy Predictive Coding implementation.")
            self.backend_type = "legacy"
            self.encoder = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            )
            self.decoder = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.output_size)
            )
            self.error_unit = nn.MSELoss(reduction='none')

    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> Any:
        """
        順伝播処理
        
        Args:
            x (torch.Tensor): 入力データ [Batch, Input_Size]
            target (torch.Tensor, optional): 教師データ（予測誤差計算用）。
                                           Noneの場合は予測値そのものを返す。
        
        Returns:
            output: 予測値、または予測誤差 (targetが与えられた場合)
        """
        
        if self.backend_type == "sara":
            # SARA Engine + Adapter を通した処理
            # Adapter内部で入力の整形やSARAのstep実行が行われる想定
            if target is not None:
                # 学習時: ターゲットとの誤差を計算して返す、または内部状態を更新
                return self.adapter(x, target=target)
            else:
                # 推論時: 予測値を返す
                return self.adapter(x)
                
        else:
            # Legacy implementation
            latent = self.encoder(x)
            prediction = self.decoder(latent)
            
            if target is not None:
                # 予測誤差を返す
                error = self.error_unit(prediction, target)
                return error.mean()
            else:
                return prediction

    def get_internal_state(self) -> Dict[str, Any]:
        """内部状態を取得（可視化用）"""
        if self.backend_type == "sara":
            return self.engine.get_state()
        else:
            return {"mode": "legacy"}

    def reset_state(self):
        """状態のリセット（時系列処理の開始時など）"""
        if self.backend_type == "sara":
            self.engine.reset()