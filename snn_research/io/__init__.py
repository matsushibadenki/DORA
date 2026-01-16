# ファイルパス: snn_research/io/__init__.py
# Title: IO Module Init
# 修正: NeuromorphicDataset -> NeuromorphicDataFactory に変更

from .spike_encoder import (
    SpikeEncoder, 
    RateEncoder, 
    LatencyEncoder, 
    DeltaEncoder, 
    DifferentiableTTFSEncoder,
    HybridTemporal8BitEncoder,
    TextSpikeEncoder
)
from .spike_decoder import SpikeDecoder
from .neuromorphic_dataset import NeuromorphicDataFactory, MockDVSGenerator
from .universal_encoder import UniversalEncoder

# 後方互換性用エイリアス (必要であれば)
NeuromorphicDataset = NeuromorphicDataFactory

__all__ = [
    "SpikeEncoder",
    "RateEncoder",
    "LatencyEncoder",
    "DeltaEncoder",
    "DifferentiableTTFSEncoder",
    "HybridTemporal8BitEncoder",
    "TextSpikeEncoder",
    "SpikeDecoder",
    "NeuromorphicDataFactory", # Updated
    "MockDVSGenerator",        # Added
    "NeuromorphicDataset",     # Kept as alias
    "UniversalEncoder",
]