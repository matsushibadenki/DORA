# snn_research/cognitive_architecture/async_brain_kernel.py
# 日本語タイトル: Async Brain Kernel (Type Fixed)
# 目的: mypyエラー(Incompatible types in assignment)の修正および型定義の強化。

import asyncio
import logging
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Any, Callable, List, Optional, Union

# Legacy / Component imports
from snn_research.cognitive_architecture.prefrontal_cortex import PrefrontalCortex
from snn_research.cognitive_architecture.hippocampus import Hippocampus
from snn_research.cognitive_architecture.motor_cortex import MotorCortex
from snn_research.cognitive_architecture.cortex import Cortex
from snn_research.cognitive_architecture.intrinsic_motivation import (
    IntrinsicMotivationSystem,
)
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork

logger = logging.getLogger(__name__)


@dataclass
class BrainEvent:
    type: str
    source: str
    payload: Any


class AsyncEventBus:
    def __init__(self) -> None:
        self.subscribers: Dict[str, List[Callable]] = {}
        self.queue: asyncio.Queue = asyncio.Queue()
        self.running = False

    def subscribe(self, topic: str, callback: Callable) -> None:
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)

    async def publish(self, event: BrainEvent) -> None:
        await self.queue.put(event)

    async def dispatch_worker(self) -> None:
        self.running = True
        logger.info("AsyncEventBus worker started.")
        while self.running:
            try:
                try:
                    event = await asyncio.wait_for(self.queue.get(), timeout=0.01)
                except asyncio.TimeoutError:
                    continue

                if event.type in self.subscribers:
                    for callback in self.subscribers[event.type]:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(event)
                            else:
                                callback(event)
                        except Exception as cb_err:
                            logger.error(
                                f"Error in subscriber callback for {event.type}: {cb_err}"
                            )

                self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dispatch_worker: {e}")


class AsyncArtificialBrain:
    """
    Sub-millisecond latency async brain kernel foundation.
    Implements metabolic constraints and asynchronous module execution.
    """

    def __init__(self, modules: Dict[str, Any], astrocyte: Any, max_workers: int = 1) -> None:
        self.modules = modules
        self.astrocyte = astrocyte
        self.max_workers = max_workers
        self.bus = AsyncEventBus()
        self.worker_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self.worker_task = asyncio.create_task(self.bus.dispatch_worker())
        logger.info("AsyncArtificialBrain started.")

    async def stop(self) -> None:
        self.bus.running = False
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        logger.info("AsyncArtificialBrain stopped.")

    async def _run_module(
        self, module_name: str, input_data: Any, output_event_type: str
    ) -> None:
        """
        Execute a cognitive module with metabolic cost check.
        """
        metabolic_cost = 0.5
        if hasattr(self.astrocyte, "consume_energy"):
            energy_available = self.astrocyte.consume_energy(metabolic_cost)
            if not energy_available:
                logger.warning(
                    f"Metabolic limit reached. Skipping module: {module_name}"
                )
                return

        if module_name in self.modules:
            try:
                module = self.modules[module_name]
                result = None

                if hasattr(module, "forward"):
                    if asyncio.iscoroutinefunction(module.forward):
                        result = await module.forward(input_data)
                    else:
                        result = module.forward(input_data)
                elif hasattr(module, "__call__"):
                    if asyncio.iscoroutinefunction(module.__call__):
                        result = await module(input_data)
                    else:
                        result = module(input_data)
                else:
                    logger.warning(f"Module {module_name} is not callable.")
                    return

                event = BrainEvent(
                    type=output_event_type, source=module_name, payload=result
                )
                await self.bus.publish(event)
            except Exception as e:
                logger.error(f"Error running module {module_name}: {e}")
        else:
            logger.error(f"Module {module_name} not found in brain registry.")

    async def process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    async def receive_input(self, input_data: Any) -> None:
        logger.debug(f"Brain received input type: {type(input_data)}")
        event = BrainEvent(type="SENSORY_INPUT", source="external", payload=input_data)
        await self.bus.publish(event)

        if "visual_cortex" in self.modules:
            await self._run_module("visual_cortex", input_data, "VISUAL_PROCESSED")


class ArtificialBrain:
    """
    Legacy synchronous Brain implementation for compatibility with existing tests.
    Combines basic cognitive modules: Cortex, PFC, Hippocampus, MotorCortex.
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        self.config = config or {}

        # 1. Initialize Components
        self.workspace = GlobalWorkspace(dim=256)
        self.motivation_system = IntrinsicMotivationSystem()
        self.astrocyte = AstrocyteNetwork()

        # 2. Main Cortices
        self.pfc: PrefrontalCortex = PrefrontalCortex(
            workspace=self.workspace,
            motivation_system=self.motivation_system,
            d_model=256,
        )

        self.hippocampus: Hippocampus = Hippocampus(
            capacity=100,
            input_dim=256,
            device="cpu",
        )

        self.motor_cortex: MotorCortex = MotorCortex(device="cpu")
        self.cortex = Cortex()

        logger.info("🧠 ArtificialBrain (Legacy Sync) initialized for testing.")

        # 3. Visual Cortex
        # [Fix] Explicitly annotate as Optional[Any] to prevent type inference conflict
        self.visual_cortex: Optional[Any] = None
        
        # Import manually to avoid circular imports at top if any
        try:
            from snn_research.models.bio.visual_cortex import VisualCortex
            self.visual_cortex = VisualCortex()
        except ImportError:
            # self.visual_cortex remains None
            pass

    def image_transform(self, image: Any) -> Any:
        from torchvision import transforms
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        return transform(image)

    def run_cognitive_cycle(self, sensory_input: Any) -> Dict[str, Any]:
        return self.process_step(sensory_input)

    def process_step(self, sensory_input: Any) -> Dict[str, Any]:
        """
        Execute one cognitive cycle (Synchronous).
        """
        logger.info("ArtificialBrain processing step.")

        # 1. Perception
        perceived = sensory_input

        # 2. Workspace Broadcast
        self.workspace.upload_to_workspace(
            source_name="sensory_input",
            content={"features": perceived}
            if isinstance(perceived, torch.Tensor)
            else {"raw": perceived},
            salience=0.8,
        )

        # 3. PFC Planning
        plan = self.pfc.plan(perceived)

        # 4. Motor Execution
        if plan is not None:
            self.motor_cortex.generate_command(plan)

        # 5. Memory Consolidation
        # Use store_episode method which is now guaranteed to exist in Hippocampus
        if isinstance(perceived, torch.Tensor):
            if perceived.numel() == 256:
                self.hippocampus.store_episode(perceived.view(-1))
            else:
                self.hippocampus.store_episode(torch.zeros(256))
        else:
            self.hippocampus.store_episode(torch.zeros(256))

        # 6. Metabolic Update
        self.astrocyte.consume_energy(1.0)

        return {"status": "processed", "input_type": str(type(sensory_input))}

    @property
    def thinking_engine(self) -> Any:
        return self.pfc

    @property
    def state(self) -> str:
        return self.astrocyte.get_diagnosis_report().get("status", "NORMAL")

    def sleep_cycle(self) -> None:
        self.astrocyte.replenish_energy(100.0)
        self.astrocyte.clear_fatigue(100.0)

    def get_brain_status(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "energy": self.astrocyte.energy,
            "fatigue": self.astrocyte.fatigue,
        }