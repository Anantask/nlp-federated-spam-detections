"""
Local Differential Privacy (LDP) Mechanism for Model Logits.
Implements bounded L2 clipping and calibrated Gaussian noise injection.
Applied on client side prior to logit transmission.
"""

import numpy as np
import torch
from typing import Tuple, Union, Optional

class GaussianLDPMechanism:
    """
    Gaussian Local Differential Privacy mechanism applied to prediction logits.
    
    1. Clips logits such that ||z_i||_2 <= C for each sample.
    2. Adds calibrated Gaussian noise N(0, (sigma * C)^2 * I).
    """
    def __init__(
        self,
        clipping_bound_C: float = 2.0,
        noise_multiplier_sigma: float = 1.0,
        random_seed: Optional[int] = 42
    ):
        if clipping_bound_C <= 0:
            raise ValueError(f"Clipping bound C must be positive, got {clipping_bound_C}")
        if noise_multiplier_sigma < 0:
            raise ValueError(f"Noise multiplier sigma must be non-negative, got {noise_multiplier_sigma}")

        self.C = float(clipping_bound_C)
        self.sigma = float(noise_multiplier_sigma)
        self.rng = np.random.RandomState(random_seed)

    def clip_logits(self, logits: np.ndarray) -> np.ndarray:
        """
        Clips each row vector in logits to maximum L2 norm C.
        """
        logits = np.asarray(logits, dtype=np.float32)
        norms = np.linalg.norm(logits, axis=1, keepdims=True)
        # Avoid division by zero
        scaling_factors = np.maximum(1.0, norms / self.C)
        clipped_logits = logits / scaling_factors
        return clipped_logits

    def add_noise(self, clipped_logits: np.ndarray) -> np.ndarray:
        """
        Adds Gaussian noise calibrated to L2 sensitivity C and noise multiplier sigma.
        Noise std = sigma * C.
        """
        if self.sigma <= 0.0:
            return clipped_logits.copy()

        noise_std = self.sigma * self.C
        noise = self.rng.normal(loc=0.0, scale=noise_std, size=clipped_logits.shape)
        return (clipped_logits + noise).astype(np.float32)

    def perturb(self, logits: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, dict]:
        """
        Full client-side LDP pipeline: L2 clipping followed by calibrated noise.
        """
        is_torch = isinstance(logits, torch.Tensor)
        if is_torch:
            raw_np = logits.detach().cpu().numpy()
        else:
            raw_np = np.asarray(logits)

        raw_norms = np.linalg.norm(raw_np, axis=1)
        clipped = self.clip_logits(raw_np)
        clipped_norms = np.linalg.norm(clipped, axis=1)
        perturbed = self.add_noise(clipped)

        diagnostics = {
            "clipping_bound_C": self.C,
            "noise_multiplier_sigma": self.sigma,
            "noise_std": self.sigma * self.C,
            "mean_raw_norm": float(np.mean(raw_norms)),
            "max_raw_norm": float(np.max(raw_norms)),
            "mean_clipped_norm": float(np.mean(clipped_norms)),
            "max_clipped_norm": float(np.max(clipped_norms)),
            "clipped_fraction": float(np.mean(raw_norms > self.C))
        }

        return perturbed, diagnostics

