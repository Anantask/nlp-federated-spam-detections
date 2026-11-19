"""
Unit tests for data acquisition, schema mapping, and text preprocessing.
"""

import pytest
import pandas as pd
from src.preprocessing import clean_comment_text
from src.data_loader import load_combined_dataset, VIDEO_SOURCE_MAP

def test_clean_comment_text_urls():
    text = "Check out my new song at https://www.youtube.com/watch?v=12345 please subscribe!"
    cleaned = clean_comment_text(text, normalize_urls=True, url_token="http_url")
    assert "http_url" in cleaned
    assert "https://" not in cleaned
    assert "subscribe" in cleaned

def test_clean_comment_text_html_entities():
    text = "I love this song &amp; video &quot;Best ever&quot; &#39;wow&#39;"
    cleaned = clean_comment_text(text)
    assert "&amp;" not in cleaned
    assert "&" in cleaned
    assert '"Best ever"' in cleaned
    assert "'wow'" in cleaned

def test_clean_comment_text_empty_and_none():
    assert clean_comment_text(None) == ""
    assert clean_comment_text("   ") == ""

def test_dataset_loading_and_counts():
    df, report = load_combined_dataset(raw_dir="data/raw", fallback_dir="data", remove_exact_duplicates=True)
    
    # Check loaded row counts
    assert report["total_rows_loaded"] == 1956
    # 3 exact duplicates removed: 1956 - 3 = 1953
    assert report["total_rows_after_dedup"] == 1953
    assert len(df) == 1953
    assert report["exact_duplicates_removed"] == 3

    # Check required columns
    expected_cols = ["comment_id", "author", "date", "raw_content", "original_class", "video_source", "label", "processed_content"]
    for col in expected_cols:
        assert col in df.columns, f"Missing required column {col}"

    # Check binary classes
    unique_labels = set(df["label"].unique())
    assert unique_labels == {0, 1}

    # Verify video sources
    sources = set(df["video_source"].unique())
    assert sources == set(VIDEO_SOURCE_MAP.values())
