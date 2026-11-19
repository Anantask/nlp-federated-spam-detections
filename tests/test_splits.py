"""
Unit tests for data splitting and federated client partitioning.
Verifies split proportions, zero-leakage, stratification, and client isolation.
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.data_split import (
    create_stratified_splits,
    create_source_held_out_split,
    get_split_manifest_summary
)
from src.client_partition import (
    partition_clients_source_based,
    partition_clients_dirichlet_non_iid,
    DEFAULT_CLIENTS
)

@pytest.fixture(scope="module")
def cleaned_df():
    data_path = "data/processed/youtube_spam_cleaned.csv"
    if not os.path.exists(data_path):
        from src.data_loader import load_combined_dataset, save_processed_dataset
        df, _ = load_combined_dataset(raw_dir="data/raw", fallback_dir="data")
        save_processed_dataset(df, data_path)
    return pd.read_csv(data_path)

def test_split_sizes_and_disjointness(cleaned_df):
    df_splits = create_stratified_splits(
        cleaned_df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        auxiliary_size=150,
        random_seed=42
    )

    assert len(df_splits) == len(cleaned_df)

    test_ids = set(df_splits[df_splits["split"] == "test"]["comment_id"])
    val_ids = set(df_splits[df_splits["split"] == "val"]["comment_id"])
    train_priv_ids = set(df_splits[df_splits["split"] == "train_private"]["comment_id"])
    train_aux_ids = set(df_splits[df_splits["split"] == "train_auxiliary"]["comment_id"])

    # Assert mutual exclusivity (ZERO leakage)
    assert len(test_ids.intersection(val_ids)) == 0
    assert len(test_ids.intersection(train_priv_ids)) == 0
    assert len(test_ids.intersection(train_aux_ids)) == 0
    assert len(val_ids.intersection(train_priv_ids)) == 0
    assert len(val_ids.intersection(train_aux_ids)) == 0
    assert len(train_priv_ids.intersection(train_aux_ids)) == 0

    # Total union equals entire dataset
    all_ids = test_ids | val_ids | train_priv_ids | train_aux_ids
    assert len(all_ids) == len(cleaned_df)

    # Check sizes (rounding allows +/- 1 sample)
    assert len(train_aux_ids) == 150
    assert abs(len(test_ids) - len(cleaned_df) * 0.15) <= 1
    assert abs(len(val_ids) - len(cleaned_df) * 0.15) <= 1

def test_stratification_balance(cleaned_df):
    df_splits = create_stratified_splits(cleaned_df, random_seed=42)
    overall_spam_pct = cleaned_df["label"].mean()

    for split in ["train_private", "train_auxiliary", "val", "test"]:
        sub = df_splits[df_splits["split"] == split]
        split_spam_pct = sub["label"].mean()
        # Stratification should keep spam ratio within 3 percentage points
        assert abs(split_spam_pct - overall_spam_pct) < 0.03, f"Split {split} spam % deviated: {split_spam_pct} vs {overall_spam_pct}"

def test_source_based_client_partitioning(cleaned_df):
    df_splits = create_stratified_splits(cleaned_df, random_seed=42)
    clients = partition_clients_source_based(df_splits)

    assert len(clients) == 5
    assigned_train_count = 0
    client_comment_ids = set()

    for cid, client in clients.items():
        assert client.client_name in DEFAULT_CLIENTS
        train_df = client.train_df
        assigned_train_count += len(train_df)
        
        # Verify comment IDs are disjoint
        c_ids = set(train_df["comment_id"])
        assert len(client_comment_ids.intersection(c_ids)) == 0
        client_comment_ids.update(c_ids)
        
        # Verify all rows belong to this client's video
        assert (train_df["video_source"] == client.client_name).all()

    # Total train samples across clients must match train_private count
    expected_train_count = len(df_splits[df_splits["split"] == "train_private"])
    assert assigned_train_count == expected_train_count

def test_dirichlet_non_iid_partitioning(cleaned_df):
    df_splits = create_stratified_splits(cleaned_df, random_seed=42)
    clients_non_iid = partition_clients_dirichlet_non_iid(df_splits, num_clients=5, alpha=0.5, random_seed=42)

    assert len(clients_non_iid) == 5
    assigned_train_count = 0
    client_comment_ids = set()

    for cid, client in clients_non_iid.items():
        train_df = client.train_df
        assigned_train_count += len(train_df)
        c_ids = set(train_df["comment_id"])
        assert len(client_comment_ids.intersection(c_ids)) == 0
        client_comment_ids.update(c_ids)

    expected_train_count = len(df_splits[df_splits["split"] == "train_private"])
    assert assigned_train_count == expected_train_count

def test_source_held_out_split(cleaned_df):
    held_out_source = "Shakira"
    df_heldout = create_source_held_out_split(cleaned_df, held_out_source=held_out_source, random_seed=42)
    
    test_sub = df_heldout[df_heldout["split"] == "test_held_out_source"]
    train_sub = df_heldout[df_heldout["split"] == "train_private"]
    
    # All test samples must be from Shakira
    assert (test_sub["video_source"] == held_out_source).all()
    # Zero samples in training should be from Shakira
    assert (train_sub["video_source"] != held_out_source).all()
