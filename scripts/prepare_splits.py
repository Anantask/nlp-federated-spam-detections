"""
Script to generate, validate, and save leakage-safe train/val/test splits
and federated client partitions for YouTube Spam Collection.
"""

import os
import sys
import yaml
import pandas as pd
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_split import (
    create_stratified_splits,
    create_source_held_out_split,
    get_split_manifest_summary,
    save_splits
)
from src.client_partition import (
    partition_clients_source_based,
    partition_clients_dirichlet_non_iid,
    generate_client_partition_report
)

def main():
    cleaned_data_path = "data/processed/youtube_spam_cleaned.csv"
    if not os.path.exists(cleaned_data_path):
        print(f"[ERROR] Cleaned dataset not found at '{cleaned_data_path}'. Running data validation first...")
        from scripts.validate_data import main as run_validation
        run_validation()

    print("[INFO] Loading cleaned dataset...")
    df = pd.read_csv(cleaned_data_path)
    print(f"[INFO] Dataset shape: {df.shape}")

    # Load data configuration
    config_path = "config/data.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    split_cfg = data_cfg.get("splitting", {})
    random_seed = split_cfg.get("random_seed", 42)
    train_ratio = split_cfg.get("train_ratio", 0.70)
    val_ratio = split_cfg.get("val_ratio", 0.15)
    test_ratio = split_cfg.get("test_ratio", 0.15)
    auxiliary_size = 150 # Auxiliary transfer set for distillation

    print(f"[INFO] Creating stratified splits (Train={train_ratio}, Val={val_ratio}, Test={test_ratio}, Aux={auxiliary_size}, Seed={random_seed})...")
    df_splits = create_stratified_splits(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        auxiliary_size=auxiliary_size,
        random_seed=random_seed
    )

    output_splits_path = "data/processed/splits_assignment.csv"
    save_splits(df_splits, output_splits_path)
    print(f"[INFO] Saved primary split assignments to '{output_splits_path}'.")

    # Display split manifest summary
    summary = get_split_manifest_summary(df_splits)
    print("\n================== SPLIT MANIFEST SUMMARY ==================")
    print(f"Total samples: {summary['total_samples']}")
    for split_name, count in summary["split_counts"].items():
        class_info = summary["per_split_class_distribution"][split_name]
        print(f"  - Split '{split_name}': {count} samples | Spam: {class_info['Spam (1)']} ({class_info['Spam_pct']}%), Ham: {class_info['Ham (0)']}")

    print("\n================ FEDERATED CLIENT PARTITIONS ================")
    # Configuration A: Source-based
    clients_a = partition_clients_source_based(df_splits)
    print("\n[Configuration A — Source-Based Partitioning (Primary)]:")
    for cid, client in clients_a.items():
        s = client.get_summary()
        print(f"  Client {cid} ({s['client_name']}): Train={s['num_train_samples']} (Spam={s['spam_count']}, Ham={s['ham_count']}, Spam_pct={round(s['spam_ratio']*100, 1)}%), Val={s['num_val_samples']}")

    # Configuration B: Controlled Non-IID
    clients_b = partition_clients_dirichlet_non_iid(df_splits, num_clients=5, alpha=0.5, random_seed=random_seed)
    print("\n[Configuration B — Controlled Non-IID (Dirichlet alpha=0.5)]:")
    for cid, client in clients_b.items():
        s = client.get_summary()
        print(f"  Client {cid} ({s['client_name']}): Train={s['num_train_samples']} (Spam={s['spam_count']}, Ham={s['ham_count']}, Spam_pct={round(s['spam_ratio']*100, 1)}%), Val={s['num_val_samples']}")

    # Out-of-domain Source-Held-Out Split
    df_heldout = create_source_held_out_split(df, held_out_source="Shakira", random_seed=random_seed)
    heldout_path = "data/processed/splits_heldout_shakira.csv"
    save_splits(df_heldout, heldout_path)
    print(f"\n[INFO] Generated Out-of-Domain Source-Held-Out split (Shakira test set) -> '{heldout_path}'.")

    print("\n[INFO] Data splitting and client partitioning completed successfully with zero test leakage.")

if __name__ == "__main__":
    main()
