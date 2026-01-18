# ファイルパス: tests/test_sleep_consolidation.py
# 修正: float精度の問題で失敗するテストを修正 (double型を使用)


import torch
import torch.nn as nn
from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator


class MockBrain(nn.Module):
    def __init__(self):
        super().__init__()
        # Use double precision for testing small updates
        self.layer = nn.Linear(10, 2).double()

    def forward(self, x, **kwargs):
        # Return mock logits
        return torch.randn(1, 2, dtype=torch.float64)


def test_synaptic_pruning():
    brain = MockBrain()
    consolidator = SleepConsolidator(substrate=brain)

    # Manually set some weights to be small (should be pruned) and large (should stay)
    with torch.no_grad():
        brain.layer.weight.data.fill_(0.1)  # Above threshold (0.05)
        brain.layer.weight.data[0, 0] = 0.01  # Below threshold

    # Initial count
    initial_active = (brain.layer.weight.data.abs() > 1e-6).sum().item()
    assert initial_active == 20  # 2x10 matrix

    # Run pruning
    pruned_count = consolidator._synaptic_pruning(threshold=0.05)

    # Check results
    assert pruned_count == 1
    assert brain.layer.weight.data[0, 0] == 0.0
    # specific check for non-pruned
    assert brain.layer.weight.data[0, 1] == 0.1


def test_synaptogenesis():
    brain = MockBrain()
    consolidator = SleepConsolidator(substrate=brain)

    # Manually set some weights to zero (candidates for creation)
    with torch.no_grad():
        brain.layer.weight.data.fill_(0.0)

    # Run synaptogenesis with high rate to ensure some creation
    created_count = consolidator._synaptogenesis(birth_rate=1.0)

    # Check results
    assert created_count > 0
    assert (brain.layer.weight.data.abs() > 0).sum().item() == created_count
