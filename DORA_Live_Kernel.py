# DORA_Live_Kernel.py
# Title: DORA Sentient Core v5.5 (Reinforcement Learning)
# Description: 
#   報酬系(NucleusAccumbens)を統合。
#   ユーザーが褒めたり叱ったりすることで、DORAの記憶の重みが変化する。
#   - "Good": 記憶が強化され、次回はより自信を持って反応する。
#   - "Bad": 記憶が弱化され、次回は反応が鈍くなる。

import sys
import os
import time
import logging
import torch
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent))

from app.containers import AppContainer
from snn_research.cognitive_architecture.language_cortex import LanguageCortex
from snn_research.cognitive_architecture.visual_cortex import VisualCortex
from snn_research.cognitive_architecture.motor_cortex import MotorCortex
from snn_research.cognitive_architecture.brocas_area import BrocasArea
from snn_research.cognitive_architecture.hippocampus import Hippocampus
from snn_research.cognitive_architecture.nucleus_accumbens import NucleusAccumbens # NEW

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(message)s')
kernel_logger = logging.getLogger("DORA_Kernel")
kernel_logger.setLevel(logging.INFO)

class DoraSentientKernel:
    def __init__(self):
        print("\n" + "="*70)
        print("⚡ DORA SENTIENT CORE v5.5: REINFORCEMENT LEARNING ONLINE")
        print("="*70)
        
        self.container = AppContainer()
        config_path = Path("configs/templates/base_config.yaml")
        if config_path.exists():
            self.container.config.from_yaml(str(config_path))
        
        self.container.config.training.paradigm.from_value("event_driven")
        self.container.config.device.from_value("cpu")
        
        self.os_kernel = self.container.neuromorphic_os()
        self.brain = self.os_kernel.brain
        self.os_kernel.boot()
        print("✅ Neuromorphic OS: ONLINE")

        self.lang_cortex = LanguageCortex(self.brain)
        self.lang_cortex.base_gain = 0.05
        self.lang_cortex.panic_gain = 0.5
        print(f"✅ Language Cortex: CONNECTED")

        self.visual_cortex = VisualCortex(self.brain)
        print(f"✅ Visual Cortex: CONNECTED")

        self.motor_cortex = MotorCortex(self.brain, threshold=12.0)
        print(f"✅ Motor Cortex: CONNECTED")

        self.brocas_area = BrocasArea(self.brain)
        print("✅ Broca's Area: CONNECTED")
        
        self.hippocampus = Hippocampus(self.brain)
        print("✅ Hippocampus: CONNECTED (Plasticity Enabled)")

        # [NEW] Nucleus Accumbens
        self.nucleus_accumbens = NucleusAccumbens(self.brain)
        print("✅ Nucleus Accumbens: CONNECTED (Dopamine Ready)")

        self.cycle_count = 0
        print("="*70 + "\n")

    def run_cycle(self, user_input: str):
        self.cycle_count += 1
        spikes = []
        input_type = "Thinking"
        trigger_content = user_input
        
        # --- 0. Reward Check (Feedback Loop) ---
        # ユーザーの入力が「報酬」かどうかを先に判定する
        reward_val = self.nucleus_accumbens.process_reward(user_input)
        if reward_val != 0.0:
            # 報酬がある場合、直前の記憶を更新してサイクル終了(学習ターン)
            self.hippocampus.update_last_memory(reward_val)
            print(f"\n🔄 Cycle {self.cycle_count:04d} | Mode: Learning")
            print(f"   🧠 Plasticity    : Memory adjusted based on feedback.")
            print("-" * 50)
            return # 学習ターンなので、新たな反応はしない

        # --- 1. Recall ---
        # "[SEE]"タグは除去して検索
        search_query = user_input.replace("[SEE]", "").strip()
        past_memory = self.hippocampus.recall(search_query)

        # --- 2. Sensation ---
        if user_input.upper().startswith("[SEE]"):
            content = user_input[5:].strip().lower()
            input_type = "Visual"
            trigger_content = f"Image of {content}"
            img = self._generate_simulated_image(content)
            
            if img:
                print(f"👁️ DORA is looking at: '{content}'")
                spikes = self.visual_cortex.process_image(img)
            else:
                print(f"⚠️ Unknown visual concept: '{content}'")
                spikes = []
        else:
            input_type = "Auditory"
            trigger_content = user_input
            spikes = self.lang_cortex.process_text(user_input if user_input else "...")

        # Activity Calculation
        avg_spikes = sum(spikes) / len(spikes) if spikes else 0.0

        # --- 3. Action ---
        motor_action = self.motor_cortex.monitor_and_act(spikes)
        
        # --- 4. Encoding ---
        # 行動した、または強い刺激の場合に記憶形成
        # (すでに記憶がある場合は、recallでlast_indexが更新されているのでOK)
        if motor_action == "ESCAPE" or avg_spikes > 15.0:
            self.hippocampus.encode_episode(trigger_content, motor_action, avg_spikes)

        # --- 5. Speech ---
        speech_input = user_input
        # 過去の記憶があり、かつ自信がある(>1.2)場合は、それについて言及する
        if past_memory:
            if past_memory['confidence'] > 1.2:
                speech_input = f"{user_input}. I am VERY sure this is dangerous! I remember {past_memory['action']}!"
            elif past_memory['confidence'] < 0.8:
                speech_input = f"{user_input}. I remember this, but maybe I was wrong last time?"

        speech = self.brocas_area.generate_response(speech_input, spikes)

        # --- 6. Display ---
        self._display_status(input_type, avg_spikes, motor_action, speech, past_memory)

    def _generate_simulated_image(self, concept):
        color_map = {
            "fire": (255, 69, 0), "blood": (139, 0, 0), "danger": (255, 0, 0),
            "sky": (135, 206, 235), "water": (0, 0, 255), "forest": (34, 139, 34),
            "grass": (0, 255, 0), "dark": (10, 10, 10), "night": (5, 5, 20)
        }
        for key in color_map:
            if key in concept:
                return Image.new('RGB', (224, 224), color=color_map[key])
        return None

    def _display_status(self, input_type, avg_spikes, action, speech, memory):
        print(f"\n🔄 Cycle {self.cycle_count:04d} | Mode: {input_type}")
        
        bar_len = int(avg_spikes * 2)
        bar = "█" * bar_len
        print(f"   🧠 Brain Activity: [{bar:<40}] {avg_spikes:.2f} spikes")
        
        if memory:
            conf_str = f"{memory['confidence']:.1f}"
            print(f"   💾 Hippocampus   : ⚠️ RECALL (Conf: {conf_str})")
        
        if action == "ESCAPE":
            print(f"   🦵 Motor Cortex : 🚨 \033[91m{action} (EMERGENCY)\033[0m")
        elif action == "ALERT":
            print(f"   🦵 Motor Cortex : 👀 {action}")
        else:
            print(f"   🦵 Motor Cortex : 💤 {action}")

        if speech:
            print(f"   🗣️ Broca's Area : \033[96m{speech}\033[0m")
        else:
            print(f"   🗣️ Broca's Area : (Silence)")
        print("-" * 50)

    def shutdown(self):
        print("\n💤 Shutting down DORA Sentient Core...")
        self.os_kernel.shutdown()
        print("👋 System Offline.")

def main():
    kernel = DoraSentientKernel()
    print("💡 INSTRUCTIONS: Train DORA!")
    print("   1. 'FIRE!' -> DORA escapes.")
    print("   2. 'Good job' -> Reinforce (+).")
    print("   3. 'Bad girl' -> Inhibit (-).")
    print("-" * 70)

    try:
        while True:
            try:
                user_in = input("\n>> You: ")
            except EOFError:
                break
            if user_in.lower() in ["exit", "quit"]:
                break
            kernel.run_cycle(user_in)
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        kernel.shutdown()

if __name__ == "__main__":
    main()