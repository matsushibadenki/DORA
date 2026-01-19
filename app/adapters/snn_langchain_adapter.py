# matsushibadenki/snn4/app/adapters/snn_langchain_adapter.py
# SNNモデルをLangChainのLLMインターフェースに適合させるアダプタ
#
# 機能:
# - LangChainのカスタムLLMとしてSNNモデルをラップする。
# - これにより、SNNをLangChainエコシステム（Chain, Agentなど）で利用可能になる。
# - ストリーミング応答をサポート (`_stream` メソッドを実装)。
# - `_stream` が `GenerationChunk` を返すように修正し、mypyエラーを解消。
# - mypyエラー修正: generateの引数の型を修正。
# 修正点: generateメソッドが返すタプル(トークン, 統計情報)を正しく処理するように修正。


from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk
from typing import Any, List, Mapping, Optional, Iterator
from app.deployment import SNNInferenceEngine


class SNNLangChainAdapter(LLM):
    """SNNInferenceEngineをラップするLangChainカスタムLLMクラス。"""

    snn_engine: SNNInferenceEngine

    @property
    def _llm_type(self) -> str:
        return "snn_breakthrough"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # _streamメソッドの結果を結合して、非ストリーミングの応答を返す
        full_response = ""
        for chunk in self._stream(prompt, stop, run_manager, **kwargs):
            full_response += chunk.text
        return full_response

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """SNNエンジンからテキストをストリーミングし、LangChainコールバックを呼び出す。"""
        # Helper to get deployment config safely
        config = getattr(self.snn_engine, "config", {})
        deployment_conf = config.get("deployment", {}) if isinstance(
            config, dict) else getattr(config, "deployment", {})
        if isinstance(deployment_conf, dict):
            max_len = deployment_conf.get("max_len", 50)
        else:
            max_len = getattr(deployment_conf, "max_len", 50)

        # SNNInferenceEngineのジェネレータを直接使用 (generate or predict)
        generator = getattr(self.snn_engine, "generate",
                            getattr(self.snn_engine, "predict", None))

        if generator:
            for chunk_text, _ in generator(prompt, max_len=max_len, stop_sequences=stop):
                chunk = GenerationChunk(text=chunk_text)
                yield chunk
                if run_manager:
                    run_manager.on_llm_new_token(chunk.text, chunk=chunk)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """モデルの識別パラメータを返す。"""
        config = getattr(self.snn_engine, "config", {})
        dep_conf = config.get("deployment", {}) if isinstance(
            config, dict) else getattr(config, "deployment", {})
        path = dep_conf.get("path") if isinstance(
            dep_conf, dict) else getattr(dep_conf, "path", "unknown")
        return {"model_path": path}
