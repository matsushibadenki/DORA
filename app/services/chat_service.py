# ファイルパス: app/services/chat_service.py
# 日本語タイトル: Chat Service Implementation v2.0
# 目的・内容:
#   ユーザーとの対話を管理し、推論エンジンの高度な応答（思考・記憶）を
#   ユーザーに届ける。

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

class ChatService:
    """
    ユーザーとの対話を管理するサービス。
    Neuromorphic OSの推論エンジン(SNNInferenceEngine)へのインターフェースとして機能する。
    """

    def __init__(self, snn_engine: Any):
        self.snn_engine = snn_engine
        logger.info("🗣️ Advanced ChatService initialized.")

    def chat(self, message: str) -> str:
        """
        ユーザーメッセージを処理し、脳からの応答を返す。
        """
        if not message:
            return "..."

        logger.info(f"📩 Message received: {message}")

        try:
            # 処理開始時刻
            start_time = time.time()
            
            # 高度な推論（思考ループ + 記憶検索）を実行
            response = self.snn_engine.generate_response(message)
            
            # 処理時間計算
            elapsed = time.time() - start_time
            logger.info(f"🧠 Reasoning completed in {elapsed:.2f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Chat processing error: {e}", exc_info=True)
            return "Thinking process interrupted (Internal Error)."