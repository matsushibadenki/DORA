# directory: snn_research/distillation
# file: knowledge_distillation_manager.py
# purpose: Knowledge Distillation Manager
# description: 知識蒸留を管理するクラス。
#              PhysicsInformedTrainer への依存を削除し、SARA Engine との連携を想定した形に修正。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

# Deprecated import removed:
# from snn_research.training.trainers.physics_informed import PhysicsInformedTrainer

class KnowledgeDistillationManager:
    """
    Manages the knowledge transfer process from a teacher model (or short-term memory)
    to a student model (long-term memory).
    """
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module, config: Any):
        self.teacher = teacher_model
        self.student = student_model
        self.config = config
        self.temperature = getattr(config, 'temperature', 2.0)
        self.alpha = getattr(config, 'alpha', 0.5)

    def distill_step(self, inputs: torch.Tensor, hard_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Executes a single distillation step.
        """
        self.teacher.eval()
        self.student.train()

        with torch.no_grad():
            teacher_logits = self.teacher(inputs)

        student_logits = self.student(inputs)

        # Distillation Loss (KL Divergence)
        soft_targets = nn.functional.softmax(teacher_logits / self.temperature, dim=-1)
        soft_prob = nn.functional.log_softmax(student_logits / self.temperature, dim=-1)
        
        distillation_loss = nn.functional.kl_div(soft_prob, soft_targets, reduction='batchmean') * (self.temperature ** 2)

        # Student Loss (Cross Entropy with hard labels)
        student_loss = 0.0
        if hard_labels is not None:
             student_loss = nn.functional.cross_entropy(student_logits, hard_labels)

        total_loss = self.alpha * student_loss + (1.0 - self.alpha) * distillation_loss

        return {
            "loss": total_loss,
            "distillation_loss": distillation_loss,
            "student_loss": student_loss
        }