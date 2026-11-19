"""
Execution Script for Federated Learning Baseline (FedAvg + BiLSTM).
Runs 5-client federated training across communication rounds,
tracks exact network communication payload, and logs test metrics.
"""

import os
import sys
import json
import yaml
import argparse
import pandas as pd
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from src.lstm_model import BiLSTMClassifier, SimpleTextTokenizer
from src.client_partition import partition_clients_source_based, partition_clients_dirichlet_non_iid
from src.federated_fedavg import FedAvgClient, FedAvgServer

def run_fedavg(partitioning="source_based", num_rounds=15, local_epochs=3, batch_size=32, lr=0.002):
    splits_path = "data/processed/splits_assignment.csv"
    if not os.path.exists(splits_path):
        from scripts.prepare_splits import main as run_splits
        run_splits()

    df_splits = pd.read_csv(splits_path)

    # Global held-out test data
    test_df = df_splits[df_splits["split"] == "test"]
    test_texts = test_df["processed_content"].tolist()
    test_labels = test_df["label"].tolist()

    # Training text for initial shared vocabulary
    train_pool = df_splits[df_splits["split"] == "train_private"]
    train_texts = train_pool["processed_content"].tolist()

    # Load configuration
    with open("config/training.yaml", "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)
    lstm_cfg = train_cfg["lstm"]

    print(f"[INFO] Initializing tokenizer with vocab_size={lstm_cfg['vocab_size']}...")
    tokenizer = SimpleTextTokenizer(vocab_size=lstm_cfg["vocab_size"])
    tokenizer.fit(train_texts)

    # Partition clients
    if partitioning == "source_based":
        print("[INFO] Partitioning clients using Configuration A (Source-Based)...")
        client_datasets = partition_clients_source_based(df_splits)
    else:
        print("[INFO] Partitioning clients using Configuration B (Controlled Non-IID Dirichlet)...")
        client_datasets = partition_clients_dirichlet_non_iid(df_splits, num_clients=5, alpha=0.5, random_seed=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device for Federated Training: {device}")

    # Build FedAvg clients
    clients = {}
    for cid, c_dataset in client_datasets.items():
        client = FedAvgClient(
            client_id=cid,
            client_name=c_dataset.client_name,
            client_dataset=c_dataset,
            tokenizer=tokenizer,
            max_seq_len=lstm_cfg["max_seq_length"],
            batch_size=batch_size,
            lr=lr,
            local_epochs=local_epochs,
            device=device
        )
        clients[cid] = client
        print(f"  Initialized {client.client_name}: {c_dataset.num_train_samples} samples")

    # Global model
    global_model = BiLSTMClassifier(
        vocab_size=lstm_cfg["vocab_size"],
        embedding_dim=lstm_cfg["embedding_dim"],
        hidden_dim=lstm_cfg["hidden_dim"],
        num_layers=lstm_cfg["num_layers"],
        bidirectional=lstm_cfg["bidirectional"],
        dropout=lstm_cfg["dropout"]
    )

    server = FedAvgServer(
        global_model=global_model,
        tokenizer=tokenizer,
        clients=clients,
        device=device
    )

    print(f"\n[INFO] Starting FedAvg for {num_rounds} communication rounds (local_epochs={local_epochs})...\n")
    results = server.run_federated_rounds(
        num_rounds=num_rounds,
        test_texts=test_texts,
        test_labels=test_labels,
        max_seq_len=lstm_cfg["max_seq_length"]
    )

    results["partitioning_mode"] = partitioning

    # Save results
    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"fedavg_{partitioning}_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[INFO] FedAvg results saved to '{out_file}'.")
    fm = results["final_metrics"]
    cs = results["communication_summary"]
    print("\n================== FEDAVG FINAL RESULTS ==================")
    print(f"Partitioning:      {partitioning}")
    print(f"Rounds:            {num_rounds}")
    print(f"Final Accuracy:    {fm['accuracy']:.4f}")
    print(f"Final Macro-F1:    {fm['macro_f1']:.4f}")
    print(f"Final Spam-F1:     {fm['spam_f1']:.4f}")
    print(f"Final ROC-AUC:     {fm['roc_auc']:.4f}")
    print(f"Total Comm (MB):   {cs['cumulative_total_mb']} MB")
    print(f"Comm/Round:        {round(cs['average_bytes_per_round'] / 1024.0, 2)} KB")
    print(f"Total Time:        {results['total_time_sec']}s")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partitioning", choices=["source_based", "controlled_non_iid"], default="source_based")
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--local_epochs", type=int, default=3)
    args = parser.parse_args()

    run_fedavg(partitioning=args.partitioning, num_rounds=args.rounds, local_epochs=args.local_epochs)
