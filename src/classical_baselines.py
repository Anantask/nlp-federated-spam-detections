"""
Classical NLP Baselines:
Model 1: TF-IDF + Logistic Regression
Model 2: TF-IDF + Linear Support Vector Machine (SVM)
Fitted strictly on training data and evaluated on held-out test data.
"""

import os
import sys
import pickle
import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.metrics import compute_classification_metrics, Timer


class ClassicalNLPBaseline:
    def __init__(
        self,
        model_type: str = "logistic_regression",
        max_features: int = 3000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        sublinear_tf: bool = True,
        C: float = 1.0,
        random_seed: int = 42
    ):
        self.model_type = model_type
        self.random_seed = random_seed
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=sublinear_tf
        )
        
        if model_type == "logistic_regression":
            self.model = LogisticRegression(
                C=C,
                max_iter=1000,
                class_weight="balanced",
                random_state=random_seed
            )
        elif model_type == "linear_svm":
            base_svm = LinearSVC(
                C=C,
                class_weight="balanced",
                random_state=random_seed,
                max_iter=2000,
                dual="auto"
            )
            # Wrap in CalibratedClassifierCV to enable calibrated probability outputs for ROC-AUC
            self.model = CalibratedClassifierCV(estimator=base_svm, cv=3)
        else:
            raise ValueError(f"Unknown classical model_type: '{model_type}'")

    def train(self, train_texts: list, train_labels: list) -> float:
        """
        Fits vectorizer and model strictly on training data.
        Returns training time in seconds.
        """
        with Timer() as t:
            X_train = self.vectorizer.fit_transform(train_texts)
            self.model.fit(X_train, train_labels)
        self.train_time = t.elapsed
        return self.train_time

    def evaluate(self, test_texts: list, test_labels: list) -> Dict[str, Any]:
        """
        Evaluates on test partition. Computes metrics, inference times, and model size.
        """
        with Timer() as t:
            X_test = self.vectorizer.transform(test_texts)
            y_pred = self.model.predict(X_test)
            y_prob = self.model.predict_proba(X_test)[:, 1]
            
        inference_time = t.elapsed
        per_sample_ms = (inference_time / len(test_texts)) * 1000.0 if len(test_texts) > 0 else 0.0

        metrics = compute_classification_metrics(test_labels, y_pred, y_prob)
        
        # Calculate in-memory model serialization size
        model_bytes = len(pickle.dumps(self.vectorizer)) + len(pickle.dumps(self.model))
        
        results = {
            "model_name": f"TF-IDF + {self.model_type.replace('_', ' ').title()}",
            "model_type": self.model_type,
            "train_time_sec": round(self.train_time, 4),
            "inference_time_sec": round(inference_time, 4),
            "inference_per_sample_ms": round(per_sample_ms, 3),
            "model_size_kb": round(model_bytes / 1024.0, 2),
            "metrics": metrics
        }
        return results
