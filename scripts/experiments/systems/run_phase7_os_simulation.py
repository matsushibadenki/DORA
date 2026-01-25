# ファイルパス: scripts/experiments/systems/run_phase7_os_simulation.py
# Title: Phase 7 Brain OS Simulation (Updated for New Scheduler)

from snn_research.cognitive_architecture.neuromorphic_scheduler import NeuromorphicScheduler, ProcessPriority
from app.containers import BrainContainer
import sys
import os
import logging
import time
from typing import Dict, Any

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', force=True)
logger = logging.getLogger("BrainOS")

def main():
    print("🧠 SNN Phase 7: Brain OS Simulation (New Scheduler)")
    container = BrainContainer()
    
    # Initialize basic components
    astrocyte = container.astrocyte_network()
    workspace = container.global_workspace()
    os_kernel = NeuromorphicScheduler(astrocyte, workspace)

    # --- Wrapper Functions to adapt inputs to tasks ---
    # Scheduler calls these callbacks directly.
    # To simulate "inputs", we will set state in the workspace or environment before stepping.

    def task_visual():
        # Read from workspace or sensor
        logger.info("👁️ Visual Cortex Executing...")
        return "Seen"

    def task_thinking():
        logger.info("🤔 Thinking Engine Executing...")
        time.sleep(0.05)
        return "Thought"

    def task_amygdala():
        logger.info("❤️ Amygdala Monitoring...")
        return "Felt"

    # Register persistent processes or ad-hoc tasks
    # Here we simulate input arriving by registering tasks dynamically
    
    # Scenario A: Routine Text
    print("\n--- Scenario A: Routine Text Processing ---")
    os_kernel.register_process("Text_Processing", ProcessPriority.NORMAL, task_thinking, energy_cost=5.0)
    os_kernel.step()

    # Scenario B: Danger
    print("\n--- Scenario B: Emergency Interrupt ---")
    os_kernel.register_process("Danger_Response", ProcessPriority.HIGH, task_amygdala, energy_cost=2.0)
    os_kernel.step()

    # Scenario C: Visual High Cost
    print("\n--- Scenario C: High-Cost Visual Processing ---")
    os_kernel.register_process("Visual_Analysis", ProcessPriority.NORMAL, task_visual, energy_cost=15.0)
    os_kernel.step()

    # Scenario D: Energy Starvation
    print("\n--- Scenario D: Energy Starvation ---")
    astrocyte.consume_energy(980.0) # Drain energy
    print(f"   📉 Energy dropped to {astrocyte.current_energy}!")
    
    # Try high cost thinking
    os_kernel.register_process("Deep_Thought", ProcessPriority.NORMAL, task_thinking, energy_cost=10.0)
    # Should be dropped or run in safety mode
    os_kernel.step()

    print("\n✅ OS Simulation Complete.")

if __name__ == "__main__":
    main()