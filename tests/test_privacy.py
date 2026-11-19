"""
Unit tests for Differential Privacy Mechanism, RDP Privacy Accounting,
Focal Loss, and Knowledge Distillation.
"""

import math
import pytest
import numpy as np
import torch
import torch.nn as nn
from src.privacy_mechanism import GaussianLDPMechanism
from src.privacy_accounting import RDPPrivacyAccountant
from src.distillation import FocalLoss, DistillationLoss

def test_ldp_clipping_and_noise():
    C = 2.0
    sigma = 0.5
    mech = GaussianLDPMechanism(clipping_bound_C=C, noise_multiplier_sigma=sigma, random_seed=42)

    # 1. Test vector exceeding bound
    large_vec = np.array([[6.0, 8.0]], dtype=np.float32) # Norm = 10.0
    clipped = mech.clip_logits(large_vec)
    clipped_norm = np.linalg.norm(clipped, axis=1)[0]
    assert np.isclose(clipped_norm, C, atol=1e-5)
    # Check direction is preserved
    assert np.isclose(clipped[0, 0] / clipped[0, 1], 6.0 / 8.0)

    # 2. Test vector within bound
    small_vec = np.array([[0.5, 0.5]], dtype=np.float32) # Norm ~ 0.707 < 2.0
    clipped_small = mech.clip_logits(small_vec)
    assert np.allclose(clipped_small, small_vec)

    # 3. Test noise distribution
    N = 50000
    zeros = np.zeros((N, 2), dtype=np.float32)
    noisy = mech.add_noise(zeros)
    expected_std = sigma * C # 0.5 * 2.0 = 1.0
    empirical_std = np.std(noisy)
    assert abs(empirical_std - expected_std) < 0.05

    # 4. Test no-noise mode
    mech_nonoise = GaussianLDPMechanism(clipping_bound_C=C, noise_multiplier_sigma=0.0)
    pert, diag = mech_nonoise.perturb(large_vec)
    assert np.allclose(pert, clipped)
    assert diag["noise_std"] == 0.0

def test_rdp_privacy_accountant():
    accountant = RDPPrivacyAccountant(clipping_bound_C=2.0, target_delta=1e-4)

    # 1. Zero noise yields infinite epsilon
    eps_zero = accountant.compute_epsilon(sigma=0.0, rounds=10)
    assert math.isinf(eps_zero)

    # 2. Monotonicity with noise multiplier (higher sigma -> lower epsilon)
    eps_sigma_05 = accountant.compute_epsilon(sigma=0.5, rounds=10)
    eps_sigma_10 = accountant.compute_epsilon(sigma=1.0, rounds=10)
    eps_sigma_20 = accountant.compute_epsilon(sigma=2.0, rounds=10)

    assert eps_sigma_05 > eps_sigma_10 > eps_sigma_20 > 0.0

    # 3. Monotonicity with communication rounds (more rounds -> higher epsilon)
    eps_r1 = accountant.compute_epsilon(sigma=1.0, rounds=1)
    eps_r10 = accountant.compute_epsilon(sigma=1.0, rounds=10)
    eps_r20 = accountant.compute_epsilon(sigma=1.0, rounds=20)

    assert eps_r1 < eps_r10 < eps_r20

    # 4. Privacy report dictionary structure
    report = accountant.generate_privacy_report(sigma=1.0, rounds=10)
    assert report["formal_dp_guarantee"] is True
    assert isinstance(report["epsilon"], float)
    assert report["clipping_bound_C"] == 2.0

def test_focal_loss():
    # 1. At gamma = 0, FocalLoss equals CrossEntropyLoss
    focal_ce = FocalLoss(gamma=0.0)
    standard_ce = nn.CrossEntropyLoss()

    logits = torch.tensor([[2.0, -1.0], [-0.5, 1.5]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)

    loss_focal_0 = focal_ce(logits, targets)
    loss_ce = standard_ce(logits, targets)
    assert torch.allclose(loss_focal_0, loss_ce, atol=1e-5)

    # 2. On well-classified examples, gamma=2.0 drastically downweights loss
    focal_g2 = FocalLoss(gamma=2.0)
    # Well-classified example: p_t ~ 0.95
    easy_logits = torch.tensor([[3.0, 0.0]], dtype=torch.float32)
    easy_target = torch.tensor([0], dtype=torch.long)

    ce_easy = standard_ce(easy_logits, easy_target)
    focal_easy = focal_g2(easy_logits, easy_target)

    # Focal loss modulation (1 - p_t)^2 ~ 0.0022 -> should be < 0.01 * ce_easy
    assert focal_easy.item() < ce_easy.item() * 0.01
    assert focal_easy.item() > 0.0

def test_distillation_loss():
    kd_loss = DistillationLoss(temperature=2.0, mode="kl_div")

    s_logits = torch.tensor([[1.0, -1.0], [0.5, 0.5]], dtype=torch.float32)
    t_logits = torch.tensor([[1.0, -1.0], [0.5, 0.5]], dtype=torch.float32)

    # Identical logits -> KL loss should be 0
    loss = kd_loss(s_logits, t_logits)
    assert abs(loss.item()) < 1e-5

    # Different logits -> positive loss
    t_different = torch.tensor([[-1.0, 1.0], [2.0, -2.0]], dtype=torch.float32)
    diff_loss = kd_loss(s_logits, t_different)
    assert diff_loss.item() > 0.0
