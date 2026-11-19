"""
Leakage-Safe Data Splitting Module.
Implements:
1. Primary Stratified Split (70% Train, 15% Validation, 15% Test) stratified by (video_source, label)
2. Safe Auxiliary Distillation Split carved strictly from training data (zero test leakage)
3. Source-Held-Out Split (e.g. Leave-One-Video-Out for out-of-domain evaluation)
Saves assignment manifests for strict reproducibility.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# Ensure project root is available
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def create_stratified_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    auxiliary_size: int = 150,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Creates leakage-safe stratified splits.
    Stratifies by composite key (video_source + label) so every video source
    and class is proportionally distributed across train, validation, and test.
    
    A portion of the training set is designated as 'train_auxiliary' for knowledge
    distillation, strictly ensuring zero test contamination.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"
    
    df = df.copy()
    # Create composite stratification key
    df["strat_key"] = df["video_source"].astype(str) + "_" + df["label"].astype(str)
    
    # 1. Split off test set (15%)
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed)
    train_val_idx, test_idx = next(sss_test.split(df, df["strat_key"]))
    
    train_val_df = df.iloc[train_val_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    # 2. Split train_val into train and validation
    val_fraction_of_train_val = val_ratio / (train_ratio + val_ratio)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction_of_train_val, random_state=random_seed)
    train_sub_idx, val_sub_idx = next(sss_val.split(train_val_df, train_val_df["strat_key"]))
    
    train_df = train_val_df.iloc[train_sub_idx].copy()
    val_df = train_val_df.iloc[val_sub_idx].copy()
    
    # 3. Carve out public auxiliary set strictly from training data
    # Auxiliary set is used by Fed-DistilDP as public transfer set
    if auxiliary_size > 0:
        if auxiliary_size >= len(train_df):
            raise ValueError(f"auxiliary_size ({auxiliary_size}) is too large for train set ({len(train_df)})")
        
        sss_aux = StratifiedShuffleSplit(
            n_splits=1, 
            test_size=auxiliary_size, 
            random_state=random_seed
        )
        private_train_idx, aux_idx = next(sss_aux.split(train_df, train_df["strat_key"]))
        
        train_private_df = train_df.iloc[private_train_idx].copy()
        train_aux_df = train_df.iloc[aux_idx].copy()
        
        train_private_df["split"] = "train_private"
        train_aux_df["split"] = "train_auxiliary"
    else:
        train_private_df = train_df.copy()
        train_private_df["split"] = "train_private"
        train_aux_df = pd.DataFrame()
        
    val_df["split"] = "val"
    test_df["split"] = "test"
    
    # Combine back into a single dataframe preserving original index order
    dfs_to_concat = [train_private_df, val_df, test_df]
    if not train_aux_df.empty:
        dfs_to_concat.append(train_aux_df)
        
    merged_df = pd.concat(dfs_to_concat, axis=0).sort_index()
    
    # Validation checks
    assert len(merged_df) == len(df), "Row count mismatch after splitting!"
    assert set(merged_df["split"].unique()) >= {"train_private", "val", "test"}
    
    return merged_df


def create_source_held_out_split(
    df: pd.DataFrame,
    held_out_source: str = "Shakira",
    val_ratio_within_train: float = 0.15,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Creates an out-of-domain evaluation split where an entire video subset
    is held out completely as the test set. The remaining 4 sources are used
    for training and validation.
    """
    df = df.copy()
    test_mask = df["video_source"] == held_out_source
    test_df = df[test_mask].copy()
    test_df["split"] = "test_held_out_source"
    
    train_val_df = df[~test_mask].copy()
    train_val_df["strat_key"] = train_val_df["video_source"].astype(str) + "_" + train_val_df["label"].astype(str)
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio_within_train, random_state=random_seed)
    train_idx, val_idx = next(sss.split(train_val_df, train_val_df["strat_key"]))
    
    train_df = train_val_df.iloc[train_idx].copy()
    train_df["split"] = "train_private"
    
    val_df = train_val_df.iloc[val_idx].copy()
    val_df["split"] = "val"
    
    combined = pd.concat([train_df, val_df, test_df], axis=0).sort_index()
    return combined


def get_split_manifest_summary(df_splits: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates summary statistics of the split assignment for reporting and diagnostics.
    """
    summary = {
        "total_samples": len(df_splits),
        "split_counts": df_splits["split"].value_counts().to_dict(),
        "per_split_class_distribution": {},
        "per_split_source_distribution": {}
    }
    
    for split_name in df_splits["split"].unique():
        sub = df_splits[df_splits["split"] == split_name]
        summary["per_split_class_distribution"][split_name] = {
            "Spam (1)": int((sub["label"] == 1).sum()),
            "Ham (0)": int((sub["label"] == 0).sum()),
            "Spam_pct": round(float((sub["label"] == 1).mean() * 100), 2)
        }
        summary["per_split_source_distribution"][split_name] = sub["video_source"].value_counts().to_dict()
        
    return summary


def save_splits(
    df_splits: pd.DataFrame,
    output_path: str = "data/processed/splits_assignment.csv"
) -> str:
    """
    Saves split manifest to disk for exact reproducibility across all models.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_splits.to_csv(out_path, index=False, encoding="utf-8")
    return str(out_path)
