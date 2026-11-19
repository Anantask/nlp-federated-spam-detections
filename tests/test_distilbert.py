"""
Unit tests for DistilBERT Sequence Classifier.
"""

import pytest
import torch
import numpy as np
from src.distilbert_model import DistilBERTSequenceClassifier

def test_distilbert_tokenizer_and_shapes():
    clf = DistilBERTSequenceClassifier(
        model_name="distilbert-base-uncased",
        max_seq_len=32,
        batch_size=2,
        device="cpu"
    )

    sample_texts = [
        "Please subscribe to my channel for more videos",
        "This song is amazing and beautiful"
    ]

    enc = clf.tokenize_texts(sample_texts)
    assert "input_ids" in enc
    assert "attention_mask" in enc
    assert enc["input_ids"].shape[0] == 2
    assert enc["input_ids"].shape[1] <= 32

    # Forward pass test
    preds, probs = clf.predict(sample_texts)
    assert len(preds) == 2
    assert probs.shape == (2, 2)
    assert (preds >= 0).all() and (preds <= 1).all()
    assert np.allclose(probs.sum(axis=1), np.ones(2), atol=1e-5)
