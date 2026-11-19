"""
Unit tests for baseline models (TF-IDF LR, TF-IDF SVM, Centralized BiLSTM) and metrics.
"""

import pytest
import numpy as np
import torch
from src.metrics import compute_classification_metrics
from src.classical_baselines import ClassicalNLPBaseline
from src.lstm_model import SimpleTextTokenizer, BiLSTMClassifier

def test_compute_classification_metrics():
    y_true = [1, 0, 1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 0, 0, 1, 1]
    y_prob = [0.9, 0.1, 0.8, 0.4, 0.2, 0.3, 0.85, 0.6]

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert 0.0 <= metrics["spam_f1"] <= 1.0
    assert 0.0 <= metrics["ham_f1"] <= 1.0
    assert metrics["roc_auc"] is not None and 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["confusion_matrix"]["tp"] + metrics["confusion_matrix"]["fn"] == 4

def test_classical_nlp_baselines():
    texts = [
        "check out my new video and subscribe please",
        "awesome song and video",
        "free money click here http://spam.com",
        "I love the music and artist",
        "subscribe to my channel now",
        "this is my favourite song ever"
    ]
    labels = [1, 0, 1, 0, 1, 0]

    for model_type in ["logistic_regression", "linear_svm"]:
        baseline = ClassicalNLPBaseline(model_type=model_type, max_features=100, min_df=1)
        train_time = baseline.train(texts, labels)
        assert train_time >= 0.0

        eval_res = baseline.evaluate(texts, labels)
        assert "metrics" in eval_res
        assert eval_res["metrics"]["accuracy"] > 0.5
        assert eval_res["model_size_kb"] > 0.0

def test_lstm_tokenizer_and_forward_pass():
    texts = [
        "subscribe to my channel",
        "i love this song",
        "free gift cards visit our site"
    ]
    tokenizer = SimpleTextTokenizer(vocab_size=100, min_freq=1)
    tokenizer.fit(texts)
    
    transformed = tokenizer.transform(texts, max_len=10)
    assert transformed.shape == (3, 10)
    assert (transformed >= 0).all()
    assert (transformed < 100).all()

    # Test BiLSTM forward pass
    model = BiLSTMClassifier(vocab_size=100, embedding_dim=16, hidden_dim=16, bidirectional=True)
    x = torch.tensor(transformed, dtype=torch.long)
    logits = model(x)
    
    assert logits.shape == (3, 2)
    probs = torch.softmax(logits, dim=1)
    assert probs.shape == (3, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-5)
