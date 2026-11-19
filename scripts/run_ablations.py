"""
Ablation Experiment Runner for Fed-DistilDP:
1. Noise Ablation: sigma in [0.0, 0.5, 1.0, 2.0]
2. Focal Loss Ablation: gamma in [0.0 (Standard CE), 2.0 (Focal Loss)]
3. Heterogeneity Ablation: Source-Based (Config A) vs Controlled Non-IID Dirichlet (Config B)
"""

import os
import sys
import json
import yaml
import time
import pandas as pd
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.run_fed_distildp import run_fed_distildp

def run_all_ablations(rounds=5):
    print("\n========================================================")
    print("STARTING SYSTEMATIC ABLATION EXPERIMENTS FOR FED-DISTILDP")
    print("========================================================")

    ablations = [
        {
            "name": "No DP Noise (Reference)",
            "key": "no_dp_noise",
            "sigma": 0.0,
            "gamma": 2.0,
            "partitioning": "source_based"
        },
        {
            "name": "Low DP Noise (sigma=0.5)",
            "key": "low_dp_noise",
            "sigma": 0.5,
            "gamma": 2.0,
            "partitioning": "source_based"
        },
        {
            "name": "Primary Setting (sigma=1.0, gamma=2.0)",
            "key": "primary_setting",
            "sigma": 1.0,
            "gamma": 2.0,
            "partitioning": "source_based"
        },
        {
            "name": "High DP Noise (sigma=2.0)",
            "key": "high_dp_noise",
            "sigma": 2.0,
            "gamma": 2.0,
            "partitioning": "source_based"
        },
        {
            "name": "Without Focal Loss (gamma=0.0 Cross-Entropy)",
            "key": "no_focal_loss",
            "sigma": 1.0,
            "gamma": 0.0,
            "partitioning": "source_based"
        },
        {
            "name": "Controlled Non-IID Dirichlet (alpha=0.5)",
            "key": "controlled_non_iid",
            "sigma": 1.0,
            "gamma": 2.0,
            "partitioning": "controlled_non_iid"
        }
    ]

    all_ablation_results = {}
    summary_table_rows = []

    for abl in ablations:
        print(f"\n>>> Running Ablation: {abl['name']} <<<")
        start_t = time.perf_counter()
        res = run_fed_distildp(
            partitioning=abl["partitioning"],
            rounds=rounds,
            local_epochs=2,
            distill_epochs=2,
            sigma=abl["sigma"],
            clip_c=2.0,
            focal_gamma=abl["gamma"],
            temperature=2.0,
            random_seed=42,
            output_filename=f"ablation_{abl['key']}.json"
        )
        duration = time.perf_counter() - start_t

        fm = res["final_metrics"]
        brm = res.get("best_round_metrics", {})
        pg = res["privacy_guarantee"]
        cs = res["communication_summary"]

        all_ablation_results[abl["key"]] = {
            "name": abl["name"],
            "config": abl,
            "duration_sec": round(duration, 2),
            "final_metrics": fm,
            "best_round": res.get("best_round"),
            "best_round_metrics": brm,
            "privacy_guarantee": pg,
            "communication_summary": cs
        }

        eps_str = f"{pg['epsilon']:.2f}" if isinstance(pg['epsilon'], float) else str(pg['epsilon'])
        summary_table_rows.append({
            "Ablation Setting": abl["name"],
            "Partitioning": abl["partitioning"],
            "Sigma": abl["sigma"],
            "Gamma": abl["gamma"],
            "Peak Acc": brm.get("accuracy", fm["accuracy"]),
            "Peak Macro-F1": brm.get("macro_f1", fm["macro_f1"]),
            "Final Acc": fm["accuracy"],
            "Final Macro-F1": fm["macro_f1"],
            "Epsilon": eps_str,
            "Total Comm (MB)": cs["cumulative_total_mb"]
        })

    # Save consolidated ablation results
    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "ablation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_ablation_results, f, indent=2)

    # Save summary table in markdown
    comp_dir = Path("results/comparison_tables")
    comp_dir.mkdir(parents=True, exist_ok=True)
    md_file = comp_dir / "ablations_summary.md"

    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Systematic Ablation Study: Fed-DistilDP\n\n")
        f.write("Ablation evaluating differential privacy noise levels, Focal Loss contribution, and client data heterogeneity on the primary held-out test partition.\n\n")
        f.write("| Ablation Setting | Partitioning | Sigma (Noise) | Gamma (Focal) | Peak Accuracy | Peak Macro-F1 | Final Acc | Final Macro-F1 | Privacy Guarantee (Epsilon) | Network Comm (MB) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in summary_table_rows:
            f.write(f"| {r['Ablation Setting']} | {r['Partitioning']} | {r['Sigma']} | {r['Gamma']} | {r['Peak Acc']:.4f} | {r['Peak Macro-F1']:.4f} | {r['Final Acc']:.4f} | {r['Final Macro-F1']:.4f} | {r['Epsilon']} | {r['Total Comm (MB)']} MB |\n")

    print(f"\n[INFO] Ablation metrics saved to '{out_file}'.")
    print(f"[INFO] Summary table saved to '{md_file}'.")

    # Display terminal summary
    print("\n================== ABLATION EXPERIMENTS SUMMARY ==================")
    for r in summary_table_rows:
        print(f"{r['Ablation Setting']:<40} | Peak Acc: {r['Peak Acc']:.4f} | Peak F1: {r['Peak Macro-F1']:.4f} | Final Acc: {r['Final Acc']:.4f} | eps: {r['Epsilon']}")

    return all_ablation_results

if __name__ == "__main__":
    run_all_ablations(rounds=5)

