# scripts/runners/run_autonomous_life_cycle.py
import logging
import torch.optim as optim
import torch
import sys
import os

# プロジェクトルートパス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# インポート
try:
    from snn_research.systems.autonomous_learning_loop import AutonomousLearningLoop
    from snn_research.systems.embodied_vlm_agent import EmbodiedVLMAgent
    from snn_research.models.transformer.spiking_vlm import SpikingVLM
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', force=True)
logger = logging.getLogger("LifeCycleRunner")

def main():
    logger.info("🚀 Starting Autonomous Life Cycle Simulation (Grand Finale)...")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Agent Setup ---
    vocab_size = 1000
    vision_config = {"type": "cnn", "hidden_dim": 512, "img_size": 64, "time_steps": 4}
    text_config = {"d_model": 512, "num_layers": 2}
    motor_config = {"action_dim": 64, "hidden_dim": 512, "action_type": "continuous"}

    try:
        logger.info("Building SpikingVLM...")
        vlm_model = SpikingVLM(
            vocab_size=vocab_size,
            vision_config=vision_config,
            text_config=text_config,
            projection_dim=512
        ).to(device)

        logger.info("Building EmbodiedVLMAgent...")
        agent = EmbodiedVLMAgent(
            vlm_model=vlm_model,
            motor_config=motor_config
        ).to(device)
        logger.info("✅ Agent built successfully.")

    except Exception as e:
        logger.error(f"Failed to init real agent: {e}. Using Mock.")
        agent = MockAgent({"hidden_dim": 512, "action_dim": 64}).to(device)

    optimizer = optim.AdamW(agent.parameters(), lr=1e-4)

    # --- Loop Setup (Fix: Pass config dict) ---
    loop_config = {
        "energy_capacity": 100.0,
        "fatigue_threshold": 50.0, # すぐに睡眠を見られるように低めに設定
        "curiosity_weight": 1.0
    }
    
    logger.info("Initializing AutonomousLearningLoop...")
    life_cycle = AutonomousLearningLoop(
        config=loop_config, # 辞書として渡す
        agent=agent,
        optimizer=optimizer,
        device=device
    )

    # --- Simulation Loop ---
    num_steps = 50
    logger.info(f"⏳ Running simulation for {num_steps} steps...")

    try:
        for step in range(num_steps):
            # Mock Input
            current_image = torch.randn(1, 3, 64, 64).to(device)
            current_text = torch.randint(0, vocab_size, (1, 10)).to(device)
            next_image = torch.randn(1, 3, 64, 64).to(device)

            # Step Execution
            status = life_cycle.step(current_image, current_text, next_image)

            mode = status["mode"]
            if mode == "wake":
                logger.info(
                    f"Step {step:03d} [☀️ WAKE]: Surprise={status.get('surprise', 0.0):.4f}, "
                    f"Reward={status.get('intrinsic_reward', 0.0):.4f}, "
                    f"Fatigue={status.get('fatigue', 0.0):.1f}/{loop_config['fatigue_threshold']}"
                )
            elif mode == "sleep":
                logger.info(
                    f"Step {step:03d} [🌙 SLEEP]: 💤 Memory Consolidation complete. Energy restored."
                )

    except Exception as e:
        logger.error(f"Error during simulation loop: {e}", exc_info=True)

    logger.info("✅ Grand Finale Simulation Complete.")

# --- Mock Agent for Fallback ---
class MockAgent(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fusion_dim = config["hidden_dim"]
        self.action_dim = config["action_dim"]
        self.mock_layer = torch.nn.Linear(10, self.fusion_dim)
        # ダミーのvlm属性とメソッド
        self.vlm = self._vlm_mock
        self.vlm.projector = type('obj', (object,), {'embed_dim': self.fusion_dim})

    def forward(self, img, txt):
        B = img.shape[0]
        return {
            "fused_context": torch.randn(B, self.fusion_dim, device=img.device),
            "action_pred": torch.randn(B, self.action_dim, device=img.device),
            "alignment_loss": torch.tensor(0.1, device=img.device, requires_grad=True),
        }

    def _vlm_mock(self, img, txt):
        B = img.shape[0]
        return {"fused_representation": torch.randn(B, self.fusion_dim, device=img.device)}

if __name__ == "__main__":
    main()