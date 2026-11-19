"""
Statistical Analysis and Hypothesis Testing Module for NLP Spam Detection.
Implements:
1. Paired Bootstrap Resampling (1,000 resamples) for confidence intervals
2. Paired McNemar Test for classifier discordance analysis
3. Metric difference confidence intervals and two-sided p-values
"""

import math
from typing import Dict, Any, List, Tuple, Callable, Optional, Union
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score

def compute_metric(y_true: np.ndarray, y_pred: np.ndarray, metric_name: str = "accuracy") -> float:
    if metric_name == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    elif metric_name == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    elif metric_name == "spam_f1":
        return float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    else:
        raise ValueError(f"Unknown metric_name: {metric_name}")

def bootstrap_confidence_interval(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    metric_name: str = "accuracy",
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Computes empirical bootstrap confidence interval for a single classifier.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_true)
    rng = np.random.RandomState(random_seed)

    base_metric = compute_metric(y_true, y_pred, metric_name)
    boot_metrics = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        boot_metrics.append(compute_metric(y_true[idx], y_pred[idx], metric_name))

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(boot_metrics, (alpha / 2.0) * 100))
    ci_upper = float(np.percentile(boot_metrics, (1.0 - alpha / 2.0) * 100))
    std_err = float(np.std(boot_metrics))

    return {
        "metric_name": metric_name,
        "observed_value": round(base_metric, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "std_error": round(std_err, 4),
        "confidence_level": confidence_level,
        "n_bootstraps": n_bootstraps
    }

def paired_bootstrap_test(
    y_true: Union[List[int], np.ndarray],
    y_pred_a: Union[List[int], np.ndarray],
    y_pred_b: Union[List[int], np.ndarray],
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    metric_name: str = "accuracy",
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Paired bootstrap hypothesis test comparing two models on the same test set.
    H0: Metric(A) - Metric(B) == 0.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_a = np.asarray(y_pred_a, dtype=int)
    y_pred_b = np.asarray(y_pred_b, dtype=int)
    n = len(y_true)
    rng = np.random.RandomState(random_seed)

    obs_a = compute_metric(y_true, y_pred_a, metric_name)
    obs_b = compute_metric(y_true, y_pred_b, metric_name)
    observed_diff = obs_a - obs_b

    diffs = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        m_a = compute_metric(y_true[idx], y_pred_a[idx], metric_name)
        m_b = compute_metric(y_true[idx], y_pred_b[idx], metric_name)
        diffs.append(m_a - m_b)

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(diffs, (alpha / 2.0) * 100))
    ci_upper = float(np.percentile(diffs, (1.0 - alpha / 2.0) * 100))

    # Empirical two-sided p-value
    # Fraction of bootstrap samples where sign of difference reverses
    if observed_diff > 0:
        p_val = 2.0 * np.mean(np.array(diffs) <= 0)
    elif observed_diff < 0:
        p_val = 2.0 * np.mean(np.array(diffs) >= 0)
    else:
        p_val = 1.0
    p_val = min(1.0, float(p_val))

    significant = bool(ci_lower > 0 or ci_upper < 0)

    return {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "metric_name": metric_name,
        "metric_a": round(obs_a, 4),
        "metric_b": round(obs_b, 4),
        "difference": round(observed_diff, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p_value": round(p_val, 4),
        "statistically_significant": significant,
        "interpretation": (
            f"The difference ({round(observed_diff, 4)}) is statistically significant at p < {round(p_val, 4)}."
            if significant else
            f"The difference ({round(observed_diff, 4)}) is not statistically significant (95% CI spans zero: [{round(ci_lower, 4)}, {round(ci_upper, 4)}])."
        )
    }

def mcnemar_test(
    y_true: Union[List[int], np.ndarray],
    y_pred_a: Union[List[int], np.ndarray],
    y_pred_b: Union[List[int], np.ndarray],
    model_a_name: str = "Model A",
    model_b_name: str = "Model B"
) -> Dict[str, Any]:
    """
    McNemar's test for comparing paired categorical predictions on test instances.
    Contingency table:
      - n00: Both correct
      - n01: A correct, B incorrect (b)
      - n10: A incorrect, B correct (c)
      - n11: Both incorrect
    Statistic: chi2 = (|b - c| - 1)^2 / (b + c)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_a = np.asarray(y_pred_a, dtype=int)
    y_pred_b = np.asarray(y_pred_b, dtype=int)

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    b = int(np.sum(correct_a & ~correct_b)) # A correct, B wrong
    c = int(np.sum(~correct_a & correct_b)) # A wrong, B correct

    total_discordant = b + c
    if total_discordant == 0:
        return {
            "model_a": model_a_name,
            "model_b": model_b_name,
            "b_a_correct_b_wrong": 0,
            "c_a_wrong_b_correct": 0,
            "chi2_statistic": 0.0,
            "p_value": 1.0,
            "significant_at_05": False,
            "audit": "Predictions are completely identical across all test examples."
        }

    # Continuity corrected chi-square statistic
    chi2 = ((abs(b - c) - 1.0) ** 2) / float(total_discordant)
    p_value = float(1.0 - stats.chi2.cdf(chi2, df=1))

    return {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "b_a_correct_b_wrong": b,
        "c_a_wrong_b_correct": c,
        "total_discordant": total_discordant,
        "chi2_statistic": round(chi2, 4),
        "p_value": round(p_value, 4),
        "significant_at_05": bool(p_value < 0.05),
        "audit": (
            f"McNemar test: chi2={round(chi2, 4)}, p={round(p_value, 4)}. "
            f"Statistically significant difference (p < 0.05)."
            if p_value < 0.05 else
            f"McNemar test: chi2={round(chi2, 4)}, p={round(p_value, 4)}. No statistically significant discordance."
        )
    }

