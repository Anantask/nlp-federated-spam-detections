"""
Execution Script for Classical and Centralized Neural Baselines:
1. TF-IDF + Logistic Regression
2. TF-IDF + Linear SVM
3. Centralized BiLSTM
Evaluated strictly on the held-out primary test set.
"""

import os
import sys
import json
import yaml
import pandas as pd
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.classical_baselines import ClassicalNLPBaseline
from src.lstm_model import CentralizedLSTMPipeline

def main():
    splits_path = "data/processed/splits_assignment.csv"
    if not os.path.exists(splits_path):
        print(f"[INFO] Split manifest not found. Running prepare_splits.py...")
        from scripts.prepare_splits import main as run_splits
        run_splits()

    print("[INFO] Loading split assignments from 'data/processed/splits_assignment.csv'...")
    df_splits = pd.read_csv(splits_path)

    # Extract splits
    train_df = df_splits[df_splits["split"] == "train_private"]
    val_df = df_splits[df_splits["split"] == "val"]
    test_df = df_splits[df_splits["split"] == "test"]

    train_texts = train_df["processed_content"].tolist()
    train_labels = train_df["label"].tolist()

    val_texts = val_df["processed_content"].tolist()
    val_labels = val_df["label"].tolist()

    test_texts = test_df["processed_content"].tolist()
    test_labels = test_df["label"].tolist()

    print(f"[INFO] Train samples: {len(train_texts)}, Val samples: {len(val_texts)}, Test samples: {len(test_texts)}")

    # Load training configs
    with open("config/training.yaml", "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)

    all_results = {}

    # -------------------------------------------------------------
    # Model 1: TF-IDF + Logistic Regression
    # -------------------------------------------------------------
    print("\n--- Training Model 1: TF-IDF + Logistic Regression ---")
    lr_cfg = train_cfg["classical_models"]["logistic_regression"]
    tfidf_cfg = train_cfg["classical_models"]["tfidf"]
    
    lr_baseline = ClassicalNLPBaseline(
        model_type="logistic_regression",
        max_features=tfidf_cfg["max_features"],
        ngram_range=tuple(tfidf_cfg["ngram_range"]),
        min_df=tfidf_cfg["min_df"],
        sublinear_tf=tfidf_cfg["sublinear_tf"],
        C=lr_cfg["C"],
        random_seed=train_cfg.get("seed", 42)
    )
    lr_baseline.train(train_texts, train_labels)
    lr_eval = lr_baseline.evaluate(test_texts, test_labels)
    all_results["tfidf_logistic_regression"] = lr_eval
    print(f"-> Accuracy: {lr_eval['metrics']['accuracy']}, Macro-F1: {lr_eval['metrics']['macro_f1']}, Spam-F1: {lr_eval['metrics']['spam_f1']}, Train Time: {lr_eval['train_time_sec']}s")

    # -------------------------------------------------------------
    # Model 2: TF-IDF + Linear SVM
    # -------------------------------------------------------------
    print("\n--- Training Model 2: TF-IDF + Linear SVM ---")
    svm_cfg = train_cfg["classical_models"]["svm"]
    svm_baseline = ClassicalNLPBaseline(
        model_type="linear_svm",
        max_features=tfidf_cfg["max_features"],
        ngram_range=tuple(tfidf_cfg["ngram_range"]),
        min_df=tfidf_cfg["min_df"],
        sublinear_tf=tfidf_cfg["sublinear_tf"],
        C=svm_cfg["C"],
        random_seed=train_cfg.get("seed", 42)
    )
    svm_baseline.train(train_texts, train_labels)
    svm_eval = svm_baseline.evaluate(test_texts, test_labels)
    all_results["tfidf_linear_svm"] = svm_eval
    print(f"-> Accuracy: {svm_eval['metrics']['accuracy']}, Macro-F1: {svm_eval['metrics']['macro_f1']}, Spam-F1: {svm_eval['metrics']['spam_f1']}, Train Time: {svm_eval['train_time_sec']}s")

    # -------------------------------------------------------------
    # Model 3: Centralized BiLSTM
    # -------------------------------------------------------------
    print("\n--- Training Model 3: Centralized BiLSTM ---")
    lstm_cfg = train_cfg["lstm"]
    lstm_pipeline = CentralizedLSTMPipeline(
        vocab_size=lstm_cfg["vocab_size"],
        embedding_dim=lstm_cfg["embedding_dim"],
        hidden_dim=lstm_cfg["hidden_dim"],
        num_layers=lstm_cfg["num_layers"],
        bidirectional=lstm_cfg["bidirectional"],
        dropout=lstm_cfg["dropout"],
        max_seq_len=lstm_cfg["max_seq_length"],
        lr=lstm_cfg["learning_rate"],
        weight_decay=float(lstm_cfg["weight_decay"]),
        batch_size=lstm_cfg["batch_size"],
        epochs=lstm_cfg["epochs"],
        patience=lstm_cfg["early_stopping_patience"],
        random_seed=train_cfg.get("seed", 42)
    )
    fit_summary = lstm_pipeline.fit(train_texts, train_labels, val_texts, val_labels)
    print(f"BiLSTM training finished in {fit_summary['train_time_sec']}s across {fit_summary['epochs_trained']} epochs. Best Val Macro-F1: {fit_summary['best_val_macro_f1']}")
    lstm_eval = lstm_pipeline.evaluate(test_texts, test_labels)
    all_results["centralized_bilstm"] = lstm_eval
    print(f"-> Accuracy: {lstm_eval['metrics']['accuracy']}, Macro-F1: {lstm_eval['metrics']['macro_f1']}, Spam-F1: {lstm_eval['metrics']['spam_f1']}, Train Time: {lstm_eval['train_time_sec']}s")

    # Save results to JSON
    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "baseline_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[INFO] Baseline results saved to '{out_json}'.")

    # Save comparison summary markdown table
    comp_dir = Path("results/comparison_tables")
    comp_dir.mkdir(parents=True, exist_ok=True)
    summary_md = comp_dir / "baselines_summary.md"
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Baseline Evaluation on YouTube Spam Collection (Primary Test Set)\n\n")
        f.write("| Model | Accuracy | Macro-F1 | Weighted-F1 | Spam Precision | Spam Recall | Spam F1 | ROC-AUC | Train Time (s) | Inference (ms/sample) | Size (KB) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for k, v in all_results.items():
            m = v["metrics"]
            roc = f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "N/A"
            f.write(f"| {v['model_name']} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['weighted_f1']:.4f} | {m['spam_precision']:.4f} | {m['spam_recall']:.4f} | {m['spam_f1']:.4f} | {roc} | {v['train_time_sec']} | {v['inference_per_sample_ms']} | {v['model_size_kb']} |\n")

    print(f"[INFO] Summary table saved to '{summary_md}'.")

    # Print summary table to stdout
    print("\n================== BASELINES SUMMARY TABLE ==================")
    for k, v in all_results.items():
        m = v["metrics"]
        print(f"Model: {v['model_name']:<28} | Acc: {m['accuracy']:.4f} | Macro-F1: {m['macro_f1']:.4f} | Spam-F1: {m['spam_f1']:.4f} | ROC-AUC: {m['roc_auc']} | Time: {v['train_time_sec']}s")

if __name__ == "__main__":
    main()
