"""
Execution Script for the Proposed Fed-DistilDP Framework:
Federated Knowledge Distillation with Local Differential Privacy
and Adaptive Class-Imbalance Compensation.
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
from src.client_partition import partition_clients_source_based, partition_clients_dirichlet_non_iid
from src.privacy_mechanism import GaussianLDPMechanism
from src.privacy_accounting import RDPPrivacyAccountant
from src.fed_distildp import FedDistilDPClient, FedDistilDPServer

def run_fed_distildp(
    partitioning="source_based",
    rounds=10,
    local_epochs=2,
    distill_epochs=2,
    sigma=1.0,
    clip_c=2.0,
    focal_gamma=2.0,
    temperature=2.0,
    random_seed=42,
    output_filename=None
):
    splits_path = "data/processed/splits_assignment.csv"
    if not os.path.exists(splits_path):
        from scripts.prepare_splits import main as run_splits
        run_splits()

    df_splits = pd.read_csv(splits_path)

    # Primary held-out test data (ZERO test leakage)
    test_df = df_splits[df_splits["split"] == "test"]
    test_texts = test_df["processed_content"].tolist()
    test_labels = test_df["label"].tolist()

    # Auxiliary public transfer dataset (strictly carved from training data)
    aux_df = df_splits[df_splits["split"] == "train_auxiliary"]
    auxiliary_texts = aux_df["processed_content"].tolist()

    # Partition clients
    if partitioning == "source_based":
        print("[INFO] Partitioning clients using Configuration A (Source-Based: 5 video clients)...")
        client_datasets = partition_clients_source_based(df_splits)
    else:
        print("[INFO] Partitioning clients using Configuration B (Controlled Non-IID Dirichlet alpha=0.5)...")
        client_datasets = partition_clients_dirichlet_non_iid(df_splits, num_clients=5, alpha=0.5, random_seed=random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device for Fed-DistilDP Training: {device}")
    print(f"[INFO] Auxiliary Transfer Set Size: {len(auxiliary_texts)} samples")
    print(f"[INFO] Privacy Settings: sigma={sigma}, clipping_bound_C={clip_c}, target_delta=1e-4")
    print(f"[INFO] Loss Settings: Focal Loss gamma={focal_gamma}, Distillation temperature={temperature}")

    # Build FedDistilDP clients
    clients = {}
    for cid, c_data in client_datasets.items():
        client = FedDistilDPClient(
            client_id=cid,
            client_name=c_data.client_name,
            client_dataset=c_data,
            max_seq_len=128,
            batch_size=16,
            lr=3e-5,
            focal_gamma=focal_gamma,
            device=device,
            freeze_backbone=True
        )
        clients[cid] = client
        print(f"  Initialized client {cid} ({client.client_name}): {c_data.num_train_samples} private samples | Class weights: {client.class_weights}")

    # LDP mechanism and privacy accountant
    ldp_mechanism = GaussianLDPMechanism(
        clipping_bound_C=clip_c,
        noise_multiplier_sigma=sigma,
        random_seed=random_seed
    )
    privacy_accountant = RDPPrivacyAccountant(
        clipping_bound_C=clip_c,
        target_delta=1e-4
    )

    # Server coordinator
    server = FedDistilDPServer(
        auxiliary_texts=auxiliary_texts,
        clients=clients,
        ldp_mechanism=ldp_mechanism,
        privacy_accountant=privacy_accountant,
        distillation_temperature=temperature,
        student_lr=5e-5,
        max_seq_len=128,
        device=device,
        freeze_student_backbone=False
    )

    print(f"\n[INFO] Starting Fed-DistilDP for {rounds} communication rounds...")
    results = server.run_distildp_rounds(
        num_rounds=rounds,
        local_epochs=local_epochs,
        test_texts=test_texts,
        test_labels=test_labels,
        distill_epochs=distill_epochs
    )

    results["partitioning_mode"] = partitioning

    # Save results
    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    if output_filename is None:
        output_filename = f"fed_distildp_{partitioning}_results.json"
    out_file = out_dir / output_filename
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    fm = results["final_metrics"]
    cs = results["communication_summary"]
    pg = results["privacy_guarantee"]

    print("\n================== FED-DISTILDP FINAL RESULTS ==================")
    print(f"Partitioning:      {partitioning}")
    print(f"Rounds:            {rounds}")
    brm = results.get("best_round_metrics", {})
    if brm:
        print(f"Peak Accuracy:     {brm.get('accuracy', 0.0):.4f} (Round {results.get('best_round')})")
        print(f"Peak Macro-F1:     {brm.get('macro_f1', 0.0):.4f}")
    print(f"Final Accuracy:    {fm['accuracy']:.4f}")
    print(f"Final Macro-F1:    {fm['macro_f1']:.4f}")
    print(f"Final Spam-F1:     {fm['spam_f1']:.4f}")
    print(f"Final ROC-AUC:     {fm['roc_auc']:.4f}")
    print(f"Privacy Guarantee: {pg['audit_statement']}")
    print(f"Total Comm (MB):   {cs['cumulative_total_mb']} MB ({cs['cumulative_total_bytes']} bytes)")
    print(f"Comm / Round:      {round(cs['average_bytes_per_round'] / 1024.0, 2)} KB")
    print(f"Total Time:        {results['total_time_sec']}s")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partitioning", choices=["source_based", "controlled_non_iid"], default="source_based")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--distill_epochs", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--clip_c", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--temp", type=float, default=2.0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    run_fed_distildp(
        partitioning=args.partitioning,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        distill_epochs=args.distill_epochs,
        sigma=args.sigma,
        clip_c=args.clip_c,
        focal_gamma=args.gamma,
        temperature=args.temp,
        output_filename=args.out
    )

