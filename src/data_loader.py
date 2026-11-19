"""
Data Loading, Schema Standardization, and Cleaning Pipeline for
the UCI YouTube Spam Collection Dataset.
"""

import os
import glob
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import yaml

from src.preprocessing import clean_comment_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# File mapping to clean source video names
VIDEO_SOURCE_MAP = {
    "Youtube01-Psy.csv": "Psy",
    "Youtube02-KatyPerry.csv": "KatyPerry",
    "Youtube03-LMFAO.csv": "LMFAO",
    "Youtube04-Eminem.csv": "Eminem",
    "Youtube05-Shakira.csv": "Shakira"
}

def load_and_standardize_single_csv(filepath: str) -> pd.DataFrame:
    """
    Loads an individual YouTube comment CSV, extracts metadata,
    standardizes column names, and validates the target schema.
    """
    filename = os.path.basename(filepath)
    df = pd.read_csv(filepath)
    
    # Identify video source from filename
    video_source = VIDEO_SOURCE_MAP.get(filename, Path(filepath).stem)
    df["video_source"] = video_source
    df["source_file"] = filename
    
    # Check column names
    col_mapping = {}
    for col in df.columns:
        upper_col = col.strip().upper()
        if upper_col in ["COMMENT_ID", "ID"]:
            col_mapping[col] = "comment_id"
        elif upper_col in ["AUTHOR", "USER"]:
            col_mapping[col] = "author"
        elif upper_col in ["DATE", "TIMESTAMP"]:
            col_mapping[col] = "date"
        elif upper_col in ["CONTENT", "TEXT", "COMMENT"]:
            col_mapping[col] = "raw_content"
        elif upper_col in ["CLASS", "TAG", "LABEL"]:
            col_mapping[col] = "original_class"

    df = df.rename(columns=col_mapping)
    
    # Ensure raw_content and original_class are present
    if "raw_content" not in df.columns or "original_class" not in df.columns:
        raise ValueError(f"File {filename} is missing essential text or class column. Found: {list(df.columns)}")

    # Map target label: 1 -> 1 (Spam), 0 -> 0 (Ham)
    df["label"] = df["original_class"].astype(int)
    df["label_name"] = df["label"].map({1: "Spam", 0: "Ham"})
    
    return df

def load_combined_dataset(
    raw_dir: str = "data/raw",
    fallback_dir: str = "data",
    remove_exact_duplicates: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads all five YouTube subsets, audits data quality, logs exact duplicate
    removals, processes comment text, and returns the cleaned combined DataFrame
    along with a comprehensive quality metrics dictionary.
    """
    raw_path = Path(raw_dir)
    fallback_path = Path(fallback_dir)
    
    csv_files = sorted(list(raw_path.glob("*.csv")))
    if not csv_files:
        csv_files = sorted(list(fallback_path.glob("*.csv")))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{raw_dir}' or '{fallback_dir}'. Run download_data.py first.")

    dfs = []
    total_raw_rows = 0
    source_stats = {}

    for fpath in csv_files:
        filename = fpath.name
        if filename not in VIDEO_SOURCE_MAP:
            continue
        sub_df = load_and_standardize_single_csv(str(fpath))
        total_raw_rows += len(sub_df)
        source_name = sub_df["video_source"].iloc[0]
        source_stats[source_name] = {
            "total_rows": len(sub_df),
            "spam_count": int((sub_df["label"] == 1).sum()),
            "ham_count": int((sub_df["label"] == 0).sum()),
            "missing_dates": int(sub_df["date"].isnull().sum()) if "date" in sub_df.columns else len(sub_df),
            "missing_authors": int(sub_df["author"].isnull().sum()) if "author" in sub_df.columns else 0,
        }
        dfs.append(sub_df)

    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Data quality diagnostics before cleaning
    initial_count = len(combined_df)
    exact_duplicates_mask = combined_df.duplicated(subset=["comment_id", "author", "date", "raw_content", "original_class"], keep="first")
    exact_duplicates_count = int(exact_duplicates_mask.sum())
    
    duplicate_rows_log = []
    if exact_duplicates_count > 0:
        dup_rows = combined_df[exact_duplicates_mask]
        for _, r in dup_rows.iterrows():
            dup_info = {
                "video_source": r["video_source"],
                "comment_id": r["comment_id"],
                "author": r["author"],
                "raw_content_preview": str(r["raw_content"])[:50],
                "label": int(r["label"])
            }
            duplicate_rows_log.append(dup_info)
            logger.info(f"Detected exact duplicate record: {dup_info}")

    # Remove exact duplicate records if enabled
    if remove_exact_duplicates and exact_duplicates_count > 0:
        combined_df = combined_df[~exact_duplicates_mask].copy().reset_index(drop=True)
        logger.info(f"Removed {exact_duplicates_count} exact duplicate rows. Dataset count changed from {initial_count} to {len(combined_df)}.")

    # Apply text preprocessing to generate processed_content
    combined_df["processed_content"] = combined_df["raw_content"].apply(clean_comment_text)

    # Check for empty comments
    empty_comments_count = int((combined_df["processed_content"].str.strip() == "").sum())

    # Build quality metrics report dictionary
    quality_report = {
        "total_rows_loaded": initial_count,
        "total_rows_after_dedup": len(combined_df),
        "exact_duplicates_removed": exact_duplicates_count,
        "duplicate_rows_details": duplicate_rows_log,
        "empty_processed_comments": empty_comments_count,
        "overall_class_distribution": {
            "Spam (1)": int((combined_df["label"] == 1).sum()),
            "Ham (0)": int((combined_df["label"] == 0).sum())
        },
        "source_video_distribution": source_stats,
        "column_names_and_dtypes": {col: str(dtype) for col, dtype in combined_df.dtypes.items()}
    }

    return combined_df, quality_report

def save_processed_dataset(
    df: pd.DataFrame,
    output_path: str = "data/processed/youtube_spam_cleaned.csv"
) -> str:
    """
    Saves the cleaned and preprocessed dataset to disk.
    """
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Cleaned dataset saved successfully to '{output_path}'.")
    return output_path
