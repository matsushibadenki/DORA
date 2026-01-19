# ファイルパス: app/containers.py
# 日本語タイトル: Dependency Injection Container
# 目的・内容:
#   アプリケーション全体の依存関係定義。
#   NeuromorphicOS、学習則、ハードウェア設定を一元管理する。
#   テストスイート(BrainContainer, TrainingContainer)への対応を追加。

import logging.config
import torch
import torch.nn as nn
from dependency_injector import containers, providers
from typing import Any

# Core Systems
from snn_research.core.neuromorphic_os import NeuromorphicOS
from snn_research.cognitive_architecture.async_brain_kernel import ArtificialBrain
from snn_research.learning_rules.stdp import STDPRule
from snn_research.learning_rules.forward_forward import ForwardForwardRule
from snn_research.io.universal_encoder import UniversalSpikeEncoder

# Services
from app.services.chat_service import ChatService
from app.services.image_classification_service import ImageClassificationService
from app.deployment import SNNInferenceEngine

# RL Components
try:
    from snn_research.agent.reinforcement_learner_agent import ReinforcementLearnerAgent
    from snn_research.rl_env.grid_world import GridWorldEnv
except ImportError:
    ReinforcementLearnerAgent = Any  # type: ignore
    GridWorldEnv = Any  # type: ignore

# Trainers
try:
    from snn_research.training.trainers.breakthrough import BreakthroughTrainer
    from snn_research.training.trainers.physics_informed import PhysicsInformedTrainer
    from snn_research.training.trainers.particle_filter import ParticleFilterTrainer
    from snn_research.training.bio_trainer import BioRLTrainer
except ImportError:
    # 防止策: テスト実行時にインポートエラーが出てもコンテナ定義自体は落ちないようにする
    BreakthroughTrainer = Any  # type: ignore
    PhysicsInformedTrainer = Any  # type: ignore
    ParticleFilterTrainer = Any  # type: ignore
    BioRLTrainer = Any  # type: ignore


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

    # --- Hardware / Device Configuration ---
    # 設定ファイルにdevice指定がない場合は自動判定を行うロジック
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

    # --- Learning Rules (Components) ---
    # 設計方針に基づき、学習則をコンポーネントとして提供
    stdp_rule = providers.Factory(
        STDPRule,
        learning_rate=config.training.biologically_plausible.stdp.learning_rate.as_(
            float),
        tau_pre=config.training.biologically_plausible.stdp.tau_trace.as_(
            float),
        tau_post=config.training.biologically_plausible.stdp.tau_trace.as_(
            float)
    )

    ff_rule = providers.Factory(
        ForwardForwardRule,
        learning_rate=0.01,
        threshold=2.0
    )

    # --- Neuromorphic OS Kernel (Core) ---
    # アプリケーションライフサイクル内で唯一のインスタンス(Singleton)
    neuromorphic_os = providers.Singleton(
        NeuromorphicOS,
        config=config.model,
        device_name=device_name_provider
    )

    # Alias for easy access
    brain = neuromorphic_os

    # --- Application Services ---
    # OSの上で動作するアプリケーションサービス群
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


class BrainContainer(AppContainer):
    """
    For Brain Tests (test_artificial_brain.py and others).
    """
    artificial_brain = providers.Factory(
        ArtificialBrain,
        config=AppContainer.config
    )

    spike_encoder = providers.Factory(
        UniversalSpikeEncoder,
        time_steps=AppContainer.config.model.time_steps.as_(int),
        output_dim=AppContainer.config.model.d_model.as_(int),
        device=AppContainer.device
    )

    # Needs access to agent container -> rag_system logic
    # Mocking agent_container for test compatibility
    class AgentContainerMock(containers.DeclarativeContainer):
        rag_system = providers.Factory(
            lambda: type('MockRAG', (), {'vector_store_path': '/tmp/mock'})()
        )

    agent_container = providers.Container(AgentContainerMock)


class TrainingContainer(AppContainer):
    """
    For Training Tests (test_smoke_all_paradigms.py).
    Needs: device, snn_model, optimizer, scheduler, various trainers.
    """

    # Mock Tokenizer
    class MockTokenizer:
        vocab_size = 1000

    tokenizer = providers.Factory(
        lambda: TrainingContainer.MockTokenizer()
    )

    snn_model = providers.Factory(
        NeuromorphicOS,
        config=AppContainer.config.model,
        device_name=AppContainer.device_name_provider
    )

    optimizer = providers.Factory(
        torch.optim.Adam,
        lr=1e-3
    )

    scheduler = providers.Factory(
        torch.optim.lr_scheduler.StepLR,
        step_size=10,
        gamma=0.1
    )

    criterion = providers.Factory(nn.CrossEntropyLoss)

    # Trainers
    standard_trainer = providers.Factory(
        BreakthroughTrainer,
        config=AppContainer.config,
        criterion=criterion,
        grad_clip_norm=AppContainer.config.training.grad_clip_norm.as_(float),
        use_amp=AppContainer.config.training.use_amp.as_(bool),
        log_dir=AppContainer.config.training.log_dir.as_(str),
        rank=0  # Default for single process
    )

    physics_informed_trainer = providers.Factory(
        PhysicsInformedTrainer,
        config=AppContainer.config,
        criterion=criterion,
        grad_clip_norm=AppContainer.config.training.grad_clip_norm.as_(float),
        use_amp=AppContainer.config.training.use_amp.as_(bool),
        log_dir=AppContainer.config.training.log_dir.as_(str),
        rank=0
    )

    # RL Components
    grid_world_env = providers.Factory(
        GridWorldEnv,
        device=AppContainer.device
    )

    reinforcement_agent = providers.Factory(
        ReinforcementLearnerAgent,
        input_size=4,
        output_size=4,
        device=AppContainer.device,
        model_config=AppContainer.config.model
    )

    bio_rl_trainer = providers.Factory(
        BioRLTrainer,
        agent=reinforcement_agent,
        env=grid_world_env
    )

    particle_filter_trainer = providers.Factory(
        ParticleFilterTrainer,
        base_model=snn_model,
        config=AppContainer.config,
        device=AppContainer.device  # Use resolved device object, not "auto" string
    )


# --- Legacy / Missing Container Mock ---
AgentContainer = Any  # type: ignore
