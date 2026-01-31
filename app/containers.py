# ファイルパス: app/containers.py
# 日本語タイトル: Dependency Injection Container (Config Fix)
# 目的・内容:
#   - ArtificialBrainにルート設定(config)を渡すように修正。

import logging.config
import torch
import torch.nn as nn
from dependency_injector import containers, providers
from typing import Any, TYPE_CHECKING

# Core Systems
from snn_research.core.neuromorphic_os import NeuromorphicOS
from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain
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
    if TYPE_CHECKING:
        from snn_research.agent.reinforcement_learner_agent import (
            ReinforcementLearnerAgent,
        )
        from snn_research.rl_env.grid_world import GridWorldEnv
    else:
        ReinforcementLearnerAgent = Any
        GridWorldEnv = Any

# Trainers
try:
    from snn_research.training.trainers.breakthrough import BreakthroughTrainer
    from snn_research.training.trainers.physics_informed import PhysicsInformedTrainer
    from snn_research.training.trainers.particle_filter import ParticleFilterTrainer
    from snn_research.training.bio_trainer import BioRLTrainer
except ImportError:
    if TYPE_CHECKING:
        # BreakthroughTrainer = Any
        # PhysicsInformedTrainer = Any
        # ParticleFilterTrainer = Any
        # BioRLTrainer = Any
        pass
    else:
        # [Fix] add # type: ignore to suppress "Cannot assign to a type"
        BreakthroughTrainer = Any  # type: ignore
        PhysicsInformedTrainer = Any  # type: ignore
        ParticleFilterTrainer = Any  # type: ignore
        BioRLTrainer = Any  # type: ignore


logger = logging.getLogger(__name__)


class AppContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    logging = providers.Resource(logging.config.dictConfig, config=config.logging)

    device_name_provider = providers.Callable(
        lambda d: d if d else "auto", d=config.device.as_(str)
    )

    device = providers.Factory(
        lambda name: torch.device(name)
        if name != "auto"
        else (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        ),
        name=device_name_provider,
    )

    # --- Learning Rules ---
    stdp_rule = providers.Factory(
        STDPRule,
        learning_rate=config.training.biologically_plausible.stdp.learning_rate.as_(
            float
        ),
        tau_pre=config.training.biologically_plausible.stdp.tau_trace.as_(float),
        tau_post=config.training.biologically_plausible.stdp.tau_trace.as_(float),
    )

    ff_rule = providers.Factory(ForwardForwardRule, learning_rate=0.01, threshold=2.0)

    # --- Core Systems ---

    # 1. The Brain (Model)
    # [Fix] config.model -> config (Pass full root config)
    artificial_brain = providers.Singleton(
        ArtificialBrain, config=config, device_name=device_name_provider
    )

    brain = artificial_brain

    # 2. The Kernel (OS)
    neuromorphic_os = providers.Singleton(
        NeuromorphicOS, brain=artificial_brain, tick_rate=10.0
    )

    # --- Application Services ---
    snn_engine = providers.Factory(
        SNNInferenceEngine, brain=artificial_brain, config=config
    )

    chat_service = providers.Factory(ChatService, snn_engine=snn_engine)

    image_service = providers.Factory(
        ImageClassificationService,
        brain=artificial_brain,
        config=config.services.vision,
    )


class BrainContainer(AppContainer):
    spike_encoder = providers.Factory(
        UniversalSpikeEncoder,
        time_steps=AppContainer.config.model.time_steps.as_(int),
        output_dim=AppContainer.config.model.d_model.as_(int),
        device=AppContainer.device,
    )

    class AgentContainerMock(containers.DeclarativeContainer):
        rag_system = providers.Factory(
            lambda: type("MockRAG", (), {"vector_store_path": "/tmp/mock"})()
        )

    agent_container = providers.Container(AgentContainerMock)


class TrainingContainer(AppContainer):
    class MockTokenizer:
        vocab_size = 1000

    tokenizer = providers.Factory(lambda: TrainingContainer.MockTokenizer())

    # [Fix] Use ArtificialBrain here, not NeuromorphicOS
    # [Fix] Pass full config
    snn_model = providers.Factory(
        ArtificialBrain,
        config=AppContainer.config,
        device_name=AppContainer.device_name_provider,
    )

    optimizer = providers.Factory(torch.optim.Adam, lr=1e-3)

    scheduler = providers.Factory(
        torch.optim.lr_scheduler.StepLR, step_size=10, gamma=0.1
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
        rank=0,
    )

    physics_informed_trainer = providers.Factory(
        PhysicsInformedTrainer,
        config=AppContainer.config,
        criterion=criterion,
        grad_clip_norm=AppContainer.config.training.grad_clip_norm.as_(float),
        use_amp=AppContainer.config.training.use_amp.as_(bool),
        log_dir=AppContainer.config.training.log_dir.as_(str),
        rank=0,
    )

    # RL Components
    grid_world_env = providers.Factory(GridWorldEnv, device=AppContainer.device)

    reinforcement_agent = providers.Factory(
        ReinforcementLearnerAgent,
        input_size=4,
        output_size=4,
        device=AppContainer.device,
        model_config=AppContainer.config.model,
    )

    bio_rl_trainer = providers.Factory(
        BioRLTrainer, agent=reinforcement_agent, env=grid_world_env
    )

    particle_filter_trainer = providers.Factory(
        ParticleFilterTrainer,
        base_model=snn_model,
        config=AppContainer.config,
        device=AppContainer.device,
    )


if not TYPE_CHECKING:
    AgentContainer = Any

if TYPE_CHECKING:
    AgentContainer = Any
