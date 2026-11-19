"""
Unit tests for Federated Learning (FedAvg + LSTM) and Communication Tracking.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from src.communication_metrics import CommunicationTracker
from src.federated_fedavg import FedAvgServer, FedAvgClient
from src.lstm_model import BiLSTMClassifier, SimpleTextTokenizer
from src.client_partition import FederatedClientDataset

def test_communication_tracker():
    tracker = CommunicationTracker()
    
    # Test model payload measurement
    sample_state = {"layer.weight": torch.randn(10, 10), "layer.bias": torch.zeros(10)}
    payload_bytes = tracker.measure_model_payload(sample_state)
    assert payload_bytes > 0

    # Record two rounds
    tracker.record_round(round_idx=1, client_uploads=[100, 100], server_downloads=[200, 200])
    tracker.record_round(round_idx=2, client_uploads=[150, 150], server_downloads=[200, 200])

    summary = tracker.get_summary()
    assert summary["total_rounds"] == 2
    assert summary["cumulative_upload_bytes"] == 500
    assert summary["cumulative_download_bytes"] == 800
    assert summary["cumulative_total_bytes"] == 1300

def test_fedavg_parameter_aggregation_math():
    # Test exact sample-size weighted averaging
    model = BiLSTMClassifier(vocab_size=50, embedding_dim=16, hidden_dim=16)
    server = FedAvgServer(global_model=model, tokenizer=SimpleTextTokenizer(), clients={})

    # Dummy weights for two clients
    t1 = torch.tensor([2.0, 4.0])
    t2 = torch.tensor([6.0, 8.0])
    
    # Equal sample sizes (50 each)
    updates_equal = [
        ({"fc.0.weight": t1}, 50),
        ({"fc.0.weight": t2}, 50)
    ]
    agg_equal = server.aggregate_weights(updates_equal)
    assert torch.allclose(agg_equal["fc.0.weight"], torch.tensor([4.0, 6.0]))

    # Unequal sample sizes: Client 1 has 10 samples (25%), Client 2 has 30 samples (75%)
    # Expected: 2.0*0.25 + 6.0*0.75 = 0.5 + 4.5 = 5.0
    # Expected: 4.0*0.25 + 8.0*0.75 = 1.0 + 6.0 = 7.0
    updates_unequal = [
        ({"fc.0.weight": t1}, 10),
        ({"fc.0.weight": t2}, 30)
    ]
    agg_unequal = server.aggregate_weights(updates_unequal)
    assert torch.allclose(agg_unequal["fc.0.weight"], torch.tensor([5.0, 7.0]))

def test_fedavg_client_local_training():
    tokenizer = SimpleTextTokenizer(vocab_size=100, min_freq=1)
    train_texts = ["subscribe now to win", "great song and music", "free money click url"]
    labels = [1, 0, 1]
    tokenizer.fit(train_texts)

    df = pd.DataFrame({"processed_content": train_texts, "label": labels})
    client_dataset = FederatedClientDataset(client_id=0, client_name="TestClient", train_df=df)

    client = FedAvgClient(
        client_id=0,
        client_name="TestClient",
        client_dataset=client_dataset,
        tokenizer=tokenizer,
        max_seq_len=16,
        batch_size=2,
        lr=0.01,
        local_epochs=1,
        device=torch.device("cpu")
    )

    model = BiLSTMClassifier(vocab_size=100, embedding_dim=16, hidden_dim=16)
    initial_weights = copy = {k: v.clone() for k, v in model.state_dict().items()}

    updated_weights, n_samples, avg_loss = client.train_local(initial_weights, model)

    assert n_samples == 3
    assert avg_loss >= 0.0
    # Check that keys match exactly
    assert set(updated_weights.keys()) == set(initial_weights.keys())
