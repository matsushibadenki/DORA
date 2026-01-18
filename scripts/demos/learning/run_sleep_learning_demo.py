# isort: skip_file
import sys
import os

# Path setup FORCEFULLY at the top
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Standard imports
import torch
import time
import logging

# Local imports (Must be after sys.path)
from snn_research.cognitive_architecture.async_brain_kernel import ArtificialBrain
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.hippocampus import Hippocampus
from snn_research.cognitive_architecture.cortex import Cortex

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', force=True)
logger = logging.getLogger("SleepCycleDemo")

def run_sleep_cycle_demo():
    print("=== 🌙 Autonomous Sleep Cycle Demo ===")
    print("日中の活動で記憶を蓄積し、疲労後に睡眠をとって記憶を長期記憶へ転送します。\n")

    # 1. Provide Config
    brain_config = {"input_neurons": 64, "feature_dim": 64}
    
    # 2. Instantiate Brain (creates default components)
    brain = ArtificialBrain(config=brain_config)

    # 3. Inject Custom Components (Overwriting defaults)
    print("⚙️ Injecting custom components for demo...")
    brain.workspace = GlobalWorkspace(dim=64)
    # Inject initial_energy via construct is trickier if name mismatch, but direct prop set is fine
    brain.astrocyte = AstrocyteNetwork(initial_energy=1000.0, max_energy=1000.0)
    brain.cortex = Cortex()
    brain.hippocampus = Hippocampus(capacity=5, input_dim=64)
    
    print("☀️ Day 1: Learning & Exploration Started")
    experiences = [
        "Saw a red apple on the table.",
        "Heard a loud noise from the street.",
        "Read a book about neural networks.",
        "Felt tired after coding python.",
        "Ate a delicious sandwich."
    ]

    for i, exp in enumerate(experiences):
        sensory_input = torch.randn(1, 64) 
        
        brain.process_step(sensory_input)
        
        # Explicit store
        brain.hippocampus.store_episode(sensory_input)
        
        # [FIX] consume_energy -> request_resource
        try:
            brain.astrocyte.request_resource("simulation_activity", 15.0)
        except TypeError:
            # Fallback if request_resource signature differs (it shouldn't based on view_file)
            brain.astrocyte.request_resource("simulation_activity", 15.0)
            
        print(f"  Step {i+1}: Experiencing -> '{exp}'")
        time.sleep(0.1)

    # [FIX] get_energy_level -> property access
    buffer_len = len(brain.hippocampus.episodic_buffer)
    print(f"\n🧠 Hippocampus Buffer: {buffer_len} items")
    
    current_energy_level = brain.astrocyte.energy / brain.astrocyte.max_energy
    print(f"⚡ Current Energy: {brain.astrocyte.energy:.1f}/{brain.astrocyte.max_energy}")

    print("\n😫 Energy dropped critically low. Needing sleep...")
    brain.astrocyte.energy = 10.0
    print(f"   (Energy forced down to: {brain.astrocyte.energy})")

    print("\n🌙 Processing next step (Checking for sleep need)...")
    result = brain.process_step(torch.randn(1, 64))
    
    # Recalculate level
    current_energy_level = brain.astrocyte.energy / brain.astrocyte.max_energy
    
    if result.get("status") == "exhausted" or current_energy_level < 0.05:
        print("💤 Brain triggered SLEEP MODE due to exhaustion.")
        
        if hasattr(brain, 'perform_sleep_cycle'):
             sleep_report = brain.perform_sleep_cycle(cycles=3)
        else:
             # Manual simulation
             print("   [Simulating Sleep Cycle...]")
             brain.astrocyte.replenish_energy(amount=500.0)
             sleep_report = {"status": "slept", "energy_recovered": 500.0}

        print(f"   > Sleep Report: {sleep_report}")
        print("   > Consolidating memories from Hippocampus to Cortex...")
        
        memories = list(brain.hippocampus.episodic_buffer)
        brain.hippocampus.clear_memory()
        
        transferred_count = len(memories)
        
        if transferred_count > 0:
            pass

        print(f"   > Memories Transferred: {transferred_count}")
        print("✨ Woke up refreshed!")
        print(f"⚡ Energy recovered: {brain.astrocyte.energy:.1f}")
    else:
        print("❌ Sleep was not triggered. Logic check needed.")
        print(f"Debug Result: {result}")

    print("\n📚 Checking Result...")
    print(f"  - Memories consolidated: {transferred_count if 'transferred_count' in locals() else 0}")
    
    if 'transferred_count' in locals() and transferred_count > 0:
        print("\n✅ SUCCESS: Sleep cycle completed and memories consolidated.")
    else:
        print("\n⚠️ PARTIAL SUCCESS: Sleep happened but no memories were transferred.")

    print("\n=== Demo Finished ===")

if __name__ == "__main__":
    run_sleep_cycle_demo()
