"""
Dataset Acquisition Script for YouTube Spam Collection
Downloads the official ZIP dataset from UCI Machine Learning Repository
if not present locally, extracts the 5 CSV subsets into data/raw/,
and ensures reproducible dataset access.
"""

import os
import shutil
import zipfile
import requests
import yaml
from pathlib import Path

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/380/youtube+spam+collection.zip"
EXPECTED_FILES = [
    "Youtube01-Psy.csv",
    "Youtube02-KatyPerry.csv",
    "Youtube03-LMFAO.csv",
    "Youtube04-Eminem.csv",
    "Youtube05-Shakira.csv",
]

def load_config(config_path="config/data.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

def download_and_extract(raw_dir="data/raw", fallback_dir="data"):
    raw_path = Path(raw_dir)
    fallback_path = Path(fallback_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    # Check if files already exist in raw_dir
    files_in_raw = [f for f in EXPECTED_FILES if (raw_path / f).exists()]
    if len(files_in_raw) == len(EXPECTED_FILES):
        print(f"[INFO] All {len(EXPECTED_FILES)} CSV files already present in '{raw_dir}'.")
        return [str(raw_path / f) for f in EXPECTED_FILES]

    # Check if files exist in fallback_dir
    files_in_fallback = [f for f in EXPECTED_FILES if (fallback_path / f).exists()]
    if len(files_in_fallback) == len(EXPECTED_FILES):
        print(f"[INFO] Found files in '{fallback_dir}'. Copying to '{raw_dir}'...")
        for f in EXPECTED_FILES:
            src = fallback_path / f
            dst = raw_path / f
            if not dst.exists():
                shutil.copy2(src, dst)
        print(f"[INFO] Successfully copied files to '{raw_dir}'.")
        return [str(raw_path / f) for f in EXPECTED_FILES]

    # Download ZIP from UCI if not found
    print(f"[INFO] Dataset not found locally. Downloading from {UCI_ZIP_URL}...")
    zip_path = raw_path / "youtube_spam_collection.zip"
    response = requests.get(UCI_ZIP_URL, stream=True, timeout=60)
    response.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"[INFO] Downloaded ZIP archive to '{zip_path}'. Extracting...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(raw_path)
    print(f"[INFO] Extracted dataset to '{raw_dir}'.")

    # Verify extracted files
    for f in EXPECTED_FILES:
        if not (raw_path / f).exists():
            raise FileNotFoundError(f"Expected file '{f}' missing after extraction.")
            
    print(f"[INFO] All {len(EXPECTED_FILES)} files verified successfully in '{raw_dir}'.")
    return [str(raw_path / f) for f in EXPECTED_FILES]

if __name__ == "__main__":
    download_and_extract()
