"""
Differential Privacy Accounting Module.
Implements Rényi Differential Privacy (RDP) accounting for the Gaussian Mechanism
composed over T communication rounds. Converts RDP guarantees to standard (epsilon, delta)-DP.
References:
- Mironov, I. (2017). "Rényi Differential Privacy". IEEE CSF.
"""

import math
from typing import Dict, Any, List, Optional
import numpy as np

class RDPPrivacyAccountant:
    """
    Rényi Differential Privacy (RDP) Accountant for Gaussian Mechanism.
    
    Sensitivity:
      Transmitted logits per sample have bounded L2 sensitivity Delta_2 = C.
    Mechanism:
      Gaussian noise with std = sigma * C.
    Single-step RDP of order alpha:
      eps_RDP(alpha) = alpha / (2 * sigma^2)
    T-round composition:
      eps_RDP_total(alpha) = T * alpha / (2 * sigma^2)
    Conversion to (eps, delta)-DP:
      eps(delta) = min_{alpha > 1} [ eps_RDP_total(alpha) + log(1/delta) / (alpha - 1) ]
    """
    def __init__(
        self,
        clipping_bound_C: float = 2.0,
        target_delta: float = 1e-4,
        candidate_orders: Optional[List[float]] = None
    ):
        self.C = float(clipping_bound_C)
        self.delta = float(target_delta)
        if candidate_orders is None:
            # Standard candidate alpha values used in literature (TF-Privacy / Opacus)
            self.orders = [
                1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5,
                5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 20.0,
                24.0, 28.0, 32.0, 48.0, 64.0, 128.0
            ]
        else:
            self.orders = [float(a) for a in candidate_orders if a > 1.0]

    def compute_rdp_order(self, sigma: float, alpha: float, rounds: int = 1) -> float:
        """Computes RDP of order alpha after rounds releases."""
        if sigma <= 0.0:
            return float("inf")
        return rounds * alpha / (2.0 * (sigma ** 2))

    def compute_epsilon(self, sigma: float, rounds: int = 1, delta: Optional[float] = None) -> float:
        """
        Calculates standard epsilon for given noise multiplier, rounds, and delta.
        """
        if sigma <= 0.0:
            return float("inf")
        
        target_delta = delta or self.delta
        if target_delta <= 0 or target_delta >= 1.0:
            raise ValueError(f"Delta must be in (0, 1), got {target_delta}")

        eps_candidates = []
        for alpha in self.orders:
            rdp = self.compute_rdp_order(sigma=sigma, alpha=alpha, rounds=rounds)
            # Conversion formula: eps = rdp + log(1/delta) / (alpha - 1)
            eps = rdp + (math.log(1.0 / target_delta) / (alpha - 1.0))
            eps_candidates.append(eps)

        # Also evaluate closed-form continuous optimum
        # alpha_opt = 1 + sigma * sqrt(2 * log(1/delta) / rounds)
        alpha_opt = 1.0 + sigma * math.sqrt(2.0 * math.log(1.0 / target_delta) / rounds)
        if alpha_opt > 1.0:
            rdp_opt = self.compute_rdp_order(sigma=sigma, alpha=alpha_opt, rounds=rounds)
            eps_opt = rdp_opt + (math.log(1.0 / target_delta) / (alpha_opt - 1.0))
            eps_candidates.append(eps_opt)

        return min(eps_candidates)

    def generate_privacy_report(
        self,
        sigma: float,
        rounds: int,
        delta: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generates full privacy guarantee disclosure dictionary.
        """
        target_delta = delta or self.delta
        epsilon = self.compute_epsilon(sigma=sigma, rounds=rounds, delta=target_delta)
        
        is_finite = not math.isinf(epsilon)
        
        return {
            "mechanism": "Gaussian Mechanism on L2-Clipped Logits",
            "formal_dp_guarantee": is_finite,
            "epsilon": round(epsilon, 4) if is_finite else "Infinity (No Noise Reference)",
            "delta": target_delta,
            "noise_multiplier_sigma": sigma,
            "clipping_bound_C": self.C,
            "num_communication_rounds_T": rounds,
            "adjacency_definition": "Bounded client logits over auxiliary dataset; replace one client partition",
            "sensitivity_assumption": f"L2 sensitivity bounded by C = {self.C}",
            "composition_accountant": "Rényi Differential Privacy (RDP) composed over T releases",
            "audit_statement": (
                f"Guarantees ({round(epsilon, 4)}, {target_delta})-DP under RDP composition across {rounds} rounds."
                if is_finite else
                "Noise multiplier sigma=0.0; no privacy guarantee provided (baseline reference)."
            )
        }

    def evaluate_privacy_grid(
        self,
        sigmas: List[float],
        rounds: int = 10,
        delta: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluates epsilon across a spectrum of noise settings.
        """
        target_delta = delta or self.delta
        grid_results = []
        for s in sigmas:
            grid_results.append(self.generate_privacy_report(sigma=s, rounds=rounds, delta=target_delta))
        return grid_results

