"""
Evaluation Metrics Module for Harmful-Content / Spam Detection.
Computes class-level and aggregate metrics, ROC-AUC, PR-AUC,
confusion matrices, and runtime statistics.
"""

import time
from typing import Dict, Any, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    classification_report
)

def compute_classification_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    y_prob: Optional[Union[np.ndarray, list]] = None,
    prefix: str = ""
) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics adhering to the research guidelines:
    - Accuracy
    - Macro-F1, Weighted-F1
    - Spam-class (1) precision, recall, F1
    - Ham-class (0) precision, recall, F1
    - Confusion Matrix [TN, FP, FN, TP]
    - ROC-AUC & PR-AUC (if probabilities are available)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    
    # Class-specific metrics (Spam = 1, Ham = 0)
    spam_p = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    spam_r = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    spam_f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    
    ham_p = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    ham_r = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    ham_f1 = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "spam_precision": round(spam_p, 4),
        "spam_recall": round(spam_r, 4),
        "spam_f1": round(spam_f1, 4),
        "ham_precision": round(ham_p, 4),
        "ham_recall": round(ham_r, 4),
        "ham_f1": round(ham_f1, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }
    
    # Optional ROC-AUC and PR-AUC
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        # If 2D probabilities passed, select positive class (index 1)
        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]
            
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
            pr_auc = float(average_precision_score(y_true, y_prob))
            metrics["roc_auc"] = round(roc_auc, 4)
            metrics["pr_auc"] = round(pr_auc, 4)
        except Exception:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    if prefix:
        return {f"{prefix}_{k}": v for k, v in metrics.items()}
    return metrics


class Timer:
    """Context manager for measuring runtime execution."""
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
