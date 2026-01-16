# ファイルパス: app/containers.py
# 日本語タイトル: Dependency Injection Container (Add 'brain' alias)
# 目的・内容:
#   アプリケーション全体の依存関係を管理するコンテナ。
#   NeuromorphicOSを 'brain' として参照できるように修正。

import logging.config
import torch
from dependency_injector import containers, providers

# Core Systems
from snn_research.core.neuromorphic_os import NeuromorphicOS
from snn_research.learning_rules.stdp import STDPRule
from snn_research.learning_rules.forward_forward import ForwardForwardRule

# Services
from app.services.chat_service import ChatService
from app.services.image_classification_service import ImageClassificationService
from app.deployment import SNNInferenceEngine

logger = logging.getLogger(__name__)

class AppContainer(containers.DeclarativeContainer):
    """
    Neuromorphic Research OS Application Container.
    """

    config = providers.Configuration()

    # --- Logging Setup ---
    logging = providers.Resource(
        logging.config.dictConfig,
        config=config.logging
    )

    # --- Device Configuration ---
    # config.device が存在しない、または None の場合に "auto" を返すロジック
    device_name_provider = providers.Callable(
        lambda d: d if d else "auto",
        d=config.device.as_(str)
    )

    device = providers.Factory(
        lambda name: torch.device(name) if name != "auto" else 
        (torch.device("cuda") if torch.cuda.is_available() else 
         torch.device("mps") if torch.backends.mps.is_available() else 
         torch.device("cpu")),
        name=device_name_provider
    )

    # --- Learning Rules ---
    stdp_rule = providers.Factory(
        STDPRule,
        learning_rate=config.training.biologically_plausible.stdp.learning_rate.as_(float),
        tau_pre=config.training.biologically_plausible.stdp.tau_trace.as_(float),
        tau_post=config.training.biologically_plausible.stdp.tau_trace.as_(float)
    )

    ff_rule = providers.Factory(
        ForwardForwardRule,
        learning_rate=0.01,
        threshold=2.0
    )

    # --- Neuromorphic OS Kernel ---
    neuromorphic_os = providers.Singleton(
        NeuromorphicOS,
        config=config.model,
        device_name=device_name_provider
    )

    # [Fix] main.py等からのアクセス用にエイリアスを設定
    brain = neuromorphic_os

    # --- Application Services ---
    snn_engine = providers.Factory(
        SNNInferenceEngine,
        brain=neuromorphic_os,
        config=config
    )

    chat_service = providers.Factory(
        ChatService,
        snn_engine=snn_engine
    )

    image_service = providers.Factory(
        ImageClassificationService,
        brain=neuromorphic_os,
        config=config.services.vision
    )