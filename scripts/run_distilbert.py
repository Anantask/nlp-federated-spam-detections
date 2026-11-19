"""
Transformer Benchmark Execution Script:
1. Centralized DistilBERT (Model 5)
2. Federated Transformer (FedNLP-inspired baseline, Model 6)
Evaluated on the held-out primary test set.
"""

import os
import sys
import json
import yaml
import time
import argparse
import pandas as pd
from pathlib import Path
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.distilbert_model import DistilBERTSequenceClassifier, DistilBertTextDataset
from src.communication_metrics import CommunicationTracker
from src.client_partition import partition_clients_source_based

def run_centralized_distilbert():
    splits_path = "data/processed/splits_assignment.csv"
    if not os.path.exists(splits_path):
        from scripts.prepare_splits import main as run_splits
        run_splits()

    df_splits = pd.read_csv(splits_path)

    train_df = df_splits[df_splits["split"] == "train_private"]
    val_df = df_splits[df_splits["split"] == "val"]
    test_df = df_splits[df_splits["split"] == "test"]

    train_texts = train_df["processed_content"].tolist()
    train_labels = train_df["label"].tolist()

    val_texts = val_df["processed_content"].tolist()
    val_labels = val_df["label"].tolist()

    test_texts = test_df["processed_content"].tolist()
    test_labels = test_df["label"].tolist()

    with open("config/training.yaml", "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)
    cfg = train_cfg["distilbert"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Starting Centralized DistilBERT Training on {device.upper()}...")
    print(f"  Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    clf = DistilBERTSequenceClassifier(
        model_name=cfg.get("model_name", "distilbert-base-uncased"),
        max_seq_len=cfg.get("max_seq_length", 128),
        batch_size=cfg.get("batch_size", 16),
        lr=float(cfg.get("learning_rate", 2e-5)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        epochs=cfg.get("epochs", 4),
        patience=2,
        device=device,
        random_seed=train_cfg.get("seed", 42)
    )

    fit_res = clf.fit(train_texts, train_labels, val_texts, val_labels)
    print(f"Training completed in {fit_res['train_time_sec']}s. Best Val Macro-F1: {fit_res['best_val_macro_f1']}")

    eval_res = clf.evaluate(test_texts, test_labels)
    eval_res["epochs_trained"] = fit_res["epochs_trained"]

    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "distilbert_centralized_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(eval_res, f, indent=2)

    m = eval_res["metrics"]
    print("\n================== CENTRALIZED DISTILBERT RESULTS ==================")
    print(f"Test Accuracy:     {m['accuracy']:.4f}")
    print(f"Test Macro-F1:     {m['macro_f1']:.4f}")
    print(f"Test Spam-F1:      {m['spam_f1']:.4f}")
    print(f"Test ROC-AUC:      {m['roc_auc']:.4f}")
    print(f"Model Size (MB):   {eval_res['model_size_mb']} MB")
    print(f"Inference Speed:   {eval_res['inference_per_sample_ms']} ms/sample")
    print(f"Training Time:     {eval_res['train_time_sec']}s")

    return eval_res

def run_federated_transformer(num_rounds=3, local_epochs=1):
    """
    Federated Transformer baseline inspired by FedNLP (Lin et al., 2022).
    Evaluates communication overhead and performance of parameter aggregation
    on transformer architectures across 5 video clients.
    """
    splits_path = "data/processed/splits_assignment.csv"
    df_splits = pd.read_csv(splits_path)

    test_df = df_splits[df_splits["split"] == "test"]
    test_texts = test_df["processed_content"].tolist()
    test_labels = test_df["label"].tolist()

    client_datasets = partition_clients_source_based(df_splits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Starting Federated Transformer (FedNLP baseline) on {device}...")
    print(f"Rounds: {num_rounds}, Local Epochs: {local_epochs}, Clients: 5")

    # Initialize global transformer
    global_clf = DistilBERTSequenceClassifier(
        model_name="distilbert-base-uncased",
        max_seq_len=128,
        batch_size=16,
        lr=2e-5,
        device=str(device)
    )

    tracker = CommunicationTracker()
    total_start = time.perf_counter()

    for r in range(1, num_rounds + 1):
        round_start = time.perf_counter()
        global_state = {k: v.cpu().clone() for k, v in global_clf.model.state_dict().items()}
        model_bytes = CommunicationTracker.measure_model_payload(global_state)

        server_downloads = [model_bytes for _ in client_datasets]
        client_uploads = []
        client_updates = []

        # Local training per client
        for cid, c_data in client_datasets.items():
            c_texts = c_data.train_df["processed_content"].tolist()
            c_labels = c_data.train_df["label"].tolist()
            c_val_texts = c_data.val_df["processed_content"].tolist()
            c_val_labels = c_data.val_df["label"].tolist()

            client_model = DistilBERTSequenceClassifier(
                model_name="distilbert-base-uncased",
                max_seq_len=128,
                batch_size=16,
                lr=3e-5,
                epochs=local_epochs,
                device=str(device)
            )
            client_model.model.load_state_dict({k: v.to(device) for k, v in global_state.items()})
            client_model.fit(c_texts, c_labels, c_val_texts, c_val_labels)

            local_weights = {k: v.cpu().clone() for k, v in client_model.model.state_dict().items()}
            client_updates.append((local_weights, c_data.num_train_samples))
            client_uploads.append(CommunicationTracker.measure_model_payload(local_weights))

        # Sample-weighted aggregation
        total_samples = sum(n for _, n in client_updates)
        agg_state = {k: torch.zeros_like(v) for k, v in client_updates[0][0].items()}
        for local_w, n in client_updates:
            weight_factor = n / total_samples
            for k in agg_state:
                agg_state[k] += local_w[k] * weight_factor

        global_clf.model.load_state_dict({k: v.to(device) for k, v in agg_state.items()})
        round_comm = tracker.record_round(r, client_uploads, server_downloads)
        
        test_eval = global_clf.evaluate(test_texts, test_labels)
        tm = test_eval["metrics"]
        print(f"FedNLP Round {r:02d}/{num_rounds:02d} | Test Acc: {tm['accuracy']:.4f} | Test Macro-F1: {tm['macro_f1']:.4f} | Comm: {round_comm['round_total_kb']} KB")

    total_time = time.perf_counter() - total_start
    final_eval = global_clf.evaluate(test_texts, test_labels)
    comm_summary = tracker.get_summary()

    results = {
        "model_name": "Federated DistilBERT (FedNLP Baseline)",
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "total_time_sec": round(total_time, 2),
        "final_metrics": final_eval["metrics"],
        "communication_summary": comm_summary
    }

    out_file = Path("results/metrics/distilbert_federated_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n================== FEDERATED DISTILBERT RESULTS ==================")
    print(f"Test Accuracy:   {final_eval['metrics']['accuracy']:.4f}")
    print(f"Test Macro-F1:   {final_eval['metrics']['macro_f1']:.4f}")
    print(f"Total Comm:      {comm_summary['cumulative_total_mb']} MB")
    print(f"Comm / Round:    {round(comm_summary['average_bytes_per_round'] / 1024.0, 2)} KB")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--federated", action="store_true", help="Run federated transformer baseline (FedNLP)")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    if args.federated:
        run_federated_transformer(num_rounds=args.rounds)
    else:
        run_centralized_distilbert()
