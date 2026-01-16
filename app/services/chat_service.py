# ファイルパス: app/services/chat_service.py
# (Neuromorphic OS対応版)

import time
from app.deployment import SNNInferenceEngine
from typing import Iterator, Tuple, List, Optional
from omegaconf import OmegaConf

class ChatService:
    def __init__(self, snn_engine: SNNInferenceEngine):
        """
        ChatServiceを初期化します。
        Args:
            snn_engine: SNN推論エンジン (NeuromorphicOSラッパー)。
        """
        self.snn_engine = snn_engine

    def stream_response(self, message: str, history: List[List[Optional[str]]]) -> Iterator[Tuple[List[List[Optional[str]]], str]]:
        """
        GradioのBlocks UIのために、チャット履歴と統計情報をストリーミング生成する。
        """
        # Configからmax_lenを取得 (辞書またはOmegaConfに対応)
        cfg = self.snn_engine.config
        if isinstance(cfg, dict):
             max_len = cfg.get("app", {}).get("max_len", 100)
        else:
             max_len = OmegaConf.select(cfg, "app.max_len", default=100)
             if max_len is None: max_len = 100
             
        max_len = int(max_len)

        prompt = ""
        for pair in history:
            user_msg = pair[0]
            bot_msg = pair[1]
            if user_msg is not None:
                prompt += f"User: {user_msg}\n"
            if bot_msg is not None:
                prompt += f"Assistant: {bot_msg}\n"
        prompt += f"User: {message}\nAssistant:"
        
        history.append([message, ""]) # プレースホルダー追加

        full_response = ""
        token_count = 0
        start_time = time.time()
        
        # SNNInferenceEngine.generate は (chunk, stats) をyieldする
        for chunk, stats in self.snn_engine.generate(prompt, max_len=max_len, stop_sequences=["User:"]):
            full_response += chunk
            token_count += 1
            history[-1][1] = full_response
            
            duration = time.time() - start_time
            total_spikes = stats.get("total_spikes", 0)
            
            # ゼロ除算回避
            if duration > 0:
                spikes_per_second = total_spikes / duration
                tokens_per_second = token_count / duration
            else:
                spikes_per_second = 0
                tokens_per_second = 0

            stats_md = f"""
            **Inference Time:** `{duration:.2f} s`
            **Tokens/Second:** `{tokens_per_second:.2f}`
            ---
            **Total Spikes:** `{total_spikes:,.0f}`
            **Spikes/Second:** `{spikes_per_second:,.0f}`
            """
            
            yield history, stats_md

        # 最終ログ
        final_stats = self.snn_engine.last_inference_stats
        print(f"Response Complete. Spikes: {final_stats.get('total_spikes', 0)}")