# Systematic Ablation Study: Fed-DistilDP

Ablation evaluating differential privacy noise levels, Focal Loss contribution, and client data heterogeneity on the primary held-out test partition.

| Ablation Setting | Partitioning | Sigma (Noise) | Gamma (Focal) | Peak Accuracy | Peak Macro-F1 | Final Acc | Final Macro-F1 | Privacy Guarantee (Epsilon) | Network Comm (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| No DP Noise (Reference) | source_based | 0.0 | 2.0 | 0.9010 | 0.9001 | 0.8294 | 0.8231 | Infinity (No Noise Reference) | 0.323 MB |
| Low DP Noise (sigma=0.5) | source_based | 0.5 | 2.0 | 0.8771 | 0.8770 | 0.7543 | 0.7466 | 29.19 | 0.323 MB |
| Primary Setting (sigma=1.0, gamma=2.0) | source_based | 1.0 | 2.0 | 0.8464 | 0.8425 | 0.6860 | 0.6847 | 12.10 | 0.323 MB |
| High DP Noise (sigma=2.0) | source_based | 2.0 | 2.0 | 0.7338 | 0.7140 | 0.5256 | 0.4410 | 5.42 | 0.323 MB |
| Without Focal Loss (gamma=0.0 Cross-Entropy) | source_based | 1.0 | 0.0 | 0.8225 | 0.8165 | 0.7304 | 0.7223 | 12.10 | 0.323 MB |
| Controlled Non-IID Dirichlet (alpha=0.5) | controlled_non_iid | 1.0 | 2.0 | 0.8498 | 0.8459 | 0.5768 | 0.5059 | 12.10 | 0.323 MB |
