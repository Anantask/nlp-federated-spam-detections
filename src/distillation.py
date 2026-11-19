"""
Distillation and Loss Module:
1. Class-Weighted Focal Loss (Adaptive Imbalance Compensation)
2. Temperature-Scaled Knowledge Distillation Loss (KL-Divergence)
3. Student Distillation Training Utilities
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict, Any, List

class FocalLoss(nn.Module):
    """
    Class-Weighted Focal Loss for Binary/Multi-class Classification.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    When gamma = 0, Focal Loss reduces exactly to standard Cross Entropy Loss.
    """
    def __init__(
        self,
        alpha: Optional[Union[float, List[float], torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        
        if alpha is not None:
            if isinstance(alpha, (float, int)):
                # Binary focal loss alpha for class 1, 1-alpha for class 0
                self.alpha = torch.tensor([1.0 - float(alpha), float(alpha)], dtype=torch.float32)
            elif isinstance(alpha, (list, np.ndarray)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            elif isinstance(alpha, torch.Tensor):
                self.alpha = alpha.float()
            else:
                self.alpha = None
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (batch_size, num_classes) raw unnormalized logits
        targets: (batch_size,) integer class indices
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        probs = torch.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = torch.pow(1.0 - pt, self.gamma)

        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * focal_weight * ce_loss
        else:
            loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class DistillationLoss(nn.Module):
    """
    Temperature-scaled Knowledge Distillation Loss.
    Measures KL-Divergence between student predictions and aggregated teacher predictions.
    
    L_KD = T^2 * KL( softmax(z_student / T) || softmax(z_teacher / T) )
    """
    def __init__(self, temperature: float = 2.0, mode: str = "kl_div"):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        self.temperature = float(temperature)
        self.mode = mode

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        student_logits: (batch_size, num_classes)
        teacher_logits: (batch_size, num_classes) - aggregated softened logits
        """
        T = self.temperature
        
        if self.mode == "kl_div":
            # Soften probabilities with temperature T
            log_p_student = F.log_softmax(student_logits / T, dim=-1)
            p_teacher = F.softmax(teacher_logits / T, dim=-1)
            # KLDivLoss expects input as log-probabilities, target as probabilities
            kl = F.kl_div(log_p_student, p_teacher, reduction="batchmean")
            # Scale by T^2 as shown by Hinton et al. to balance gradients
            return kl * (T ** 2)
            
        elif self.mode == "mse":
            return F.mse_loss(student_logits / T, teacher_logits / T)
        else:
            raise ValueError(f"Unknown distillation loss mode: {self.mode}")

