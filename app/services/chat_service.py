# ファイルパス: app/services/chat_service.py
# 日本語タイトル: Chat Service Implementation
# 目的・内容:
#   ユーザーからのテキストメッセージを受け取り、SNN推論エンジンに渡して
#   応答を取得するアプリケーションサービス。

import logging
from typing import Any

logger = logging.getLogger(__name__)

class ChatService:
    """
    ユーザーとの対話を管理するサービス。
    Neuromorphic OSの推論エンジン(SNNInferenceEngine)へのインターフェースとして機能する。
    """

    def __init__(self, snn_engine: Any):
        # 循環参照を避けるため型ヒントはAnyにしているが、実際は SNNInferenceEngine
        self.snn_engine = snn_engine
        logger.info("🗣️ ChatService initialized.")

    def chat(self, message: str) -> str:
        """
        ユーザーメッセージを処理し、脳からの応答を返す。
        
        Args:
            message (str): ユーザー入力テキスト
            
        Returns:
            str: エージェントの応答テキスト
        """
        if not message:
            return "..."

        logger.info(f"📩 Message received: {message}")

        try:
            # SNN推論エンジンを使用して応答を生成
            # (generate_response メソッドは app/deployment.py で定義)
            response = self.snn_engine.generate_response(message)
            return response
        except Exception as e:
            logger.error(f"❌ Chat processing error: {e}")
            return "Thinking process interrupted (Internal Error)."