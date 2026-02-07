# snn_research/scenarios/brain_v14.py
# Title: Brain V14 Scenario (Type Fixed)
# Description: mypyエラー回避版

from app.containers import BrainContainer
import os
import time
import logging
from typing import cast, Any
import torch
from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain

logger = logging.getLogger("Scenario_BrainV14")

def run_scenario(config_path: str = "configs/experiments/brain_v14_config.yaml"):
    print("\n" + "="*60)
    print("🧠 SNN Artificial Brain v14.0: Neuro-Symbolic Evolution")
    print("="*60)
    print("   Initializing Neuromorphic OS...")

    container = BrainContainer()

    if os.path.exists(config_path):
        container.config.from_yaml(config_path)
    else:
        logger.warning(f"Config file {config_path} not found. Using defaults.")
        container.config.from_dict({
            "model": {
                "architecture_type": "sformer",
                "d_model": 128,
                "time_steps": 1,
                "neuron": {"type": "scale_and_fire", "base_threshold": 4.0}
            },
            "training": {
                "biologically_plausible": {
                    "learning_rule": "CAUSAL_TRACE_V2",
                    "neuron": {"type": "lif"}
                }
            }
        })

    rag = container.agent_container.rag_system()
    kb_size = len(rag.knowledge_base)
    logger.info(f"   - RAG System initialized. Current Knowledge Base Size: {kb_size}")

    # [Fix] Explicit typing to Any to avoid "Tensor not callable" inference error
    brain: Any = container.artificial_brain()

    engine_name = "unknown"
    if hasattr(brain, 'thinking_engine'):
        engine = brain.thinking_engine
        if hasattr(engine, 'config'):
            cfg = getattr(engine, 'config')
            if isinstance(cfg, dict):
                engine_name = cfg.get("architecture_type", "unknown")
            else:
                engine_name = "custom_module"
        else:
            engine_name = engine.__class__.__name__

    print(f"   - Thinking Engine: {engine_name} (Ready)")

    astro_energy = 0.0
    if hasattr(brain, 'astrocyte'):
        astrocyte = brain.astrocyte
        astro_energy = float(astrocyte.current_energy)

    print(f"   - Astrocyte: Energy={astro_energy:.1f}")

    print("\n🌞 [Phase 1: Knowledge Acquisition]")
    dialogue = [
        "SNN stands for Spiking Neural Network.",
        "SNN uses spikes for energy efficiency.",
        "The brain sleeps to consolidate memory.",
        "Generative replay happens during sleep."
    ]

    for txt in dialogue:
        print(f"   👤 Input: '{txt}'")
        if hasattr(brain, 'run_cognitive_cycle'):
            result = brain.run_cognitive_cycle(txt)
        else:
            result = brain.process_step(txt)

        executed = result.get("executed_modules", [])
        print(f"      -> Brain processed via: {executed}")
        if result.get("consciousness"):
            print(f"      -> Consciousness: {result['consciousness']}")

        time.sleep(0.5)

    print("\n🔥 [Phase 2: High Cognitive Load]")
    print("   Simulating complex reasoning tasks to drain energy...")

    for i in range(5):
        msg = f"Complex reasoning task {i}: Calculate optimal path."
        if hasattr(brain, 'run_cognitive_cycle'):
            brain.run_cognitive_cycle(msg)
        else:
            brain.process_step(msg)

        current_energy = 0.0
        current_fatigue = 0.0
        if hasattr(brain, 'astrocyte'):
            astrocyte = brain.astrocyte
            current_energy = float(astrocyte.current_energy)
            current_fatigue = float(astrocyte.fatigue)

        print(f"   Task {i+1}: Energy {current_energy:.1f} | Fatigue {current_fatigue:.1f}")

    print("\n💤 [Phase 3: Sleep & Consolidation]")
    if brain.state != "SLEEPING":
        print("   Forcing sleep cycle due to roadmap schedule...")
        if hasattr(brain, 'sleep_cycle'):
            brain.sleep_cycle() # type: ignore
        else:
            brain.sleep()

    print("\n🌞 [Phase 4: Awakening & Evolution Check]")
    query = "SNN"
    print(f"   🧠 Checking Long-Term Memory for '{query}':")

    if hasattr(brain, 'retrieve_knowledge'):
        knowledge = brain.retrieve_knowledge(query) # type: ignore
        if not knowledge:
            print("      (No knowledge retrieved directly from Cortex retrieval)")
        else:
            for k in knowledge[:3]:
                print(f"      - {k}")

    print("\n🎉 Simulation Complete. The Artificial Brain has successfully evolved.")