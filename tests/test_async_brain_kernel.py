# ファイルパス: tests/test_async_brain_kernel.py
import unittest
import asyncio
from unittest.mock import MagicMock
from snn_research.cognitive_architecture.async_brain_kernel import AsyncArtificialBrain

class TestAsyncBrainKernel(unittest.TestCase):
    def setUp(self):
        # 既存のセットアップ（他のテスト用）
        self.astrocyte = MagicMock()
        self.astrocyte.check_metabolic_limit.return_value = True

    def test_initialization(self):
        """初期化のテスト"""
        brain = AsyncArtificialBrain(modules={}, astrocyte=self.astrocyte)
        self.assertIsNotNone(brain)

    def test_module_execution(self):
        """モジュール実行のテスト"""
        class MockModule:
            def forward(self, x):
                return f"processed_{x}"
        
        # テスト用に代謝チェックを確実にパスするモックを作成
        mock_astrocyte = MagicMock()
        mock_astrocyte.check_metabolic_limit.return_value = True
        
        brain = AsyncArtificialBrain(
            modules={"test_module": MockModule()},
            astrocyte=mock_astrocyte
        )
        
        async def run_kernel_test():
            await brain.start()
            
            # イベントバス経由ではなく、内部メソッドを直接テスト
            result_event = None
            
            # イベントリスナーをモック
            async def result_catcher(event):
                nonlocal result_event
                result_event = event
                
            brain.bus.subscribe("OUTPUT_EVENT", result_catcher)
            
            # モジュール実行
            # check_metabolic_limitがTrueを返すため、ここで実行がスキップされなくなる
            await brain._run_module("test_module", "input", "OUTPUT_EVENT")
            
            await asyncio.sleep(0.1)
            await brain.stop()
            return result_event
        
        result = asyncio.run(run_kernel_test())
        self.assertIsNotNone(result)