# Statistical Significance Analysis (Paired Bootstrap & McNemar Tests)

All statistical tests evaluated across paired held-out primary test predictions ($N=293$) with 1,000 bootstrap resamples at $\alpha = 0.05$.

| Model Comparison | Acc (A) | Acc (B) | Difference | 95% Confidence Interval | Bootstrap p-value | McNemar $\chi^2$ | McNemar p-value | Significant (p < 0.05) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Centralized DistilBERT vs. TF-IDF + Logistic Regression | 0.9863 | 0.9761 | +0.0102 | [-0.0102, 0.0307] | 0.4780 | 0.3636 | 0.5465 | No |
| Centralized DistilBERT vs. Centralized BiLSTM | 0.9863 | 0.9659 | +0.0205 | [0.0067, 0.0375] | 0.0060 | 4.1667 | 0.0412 | **Yes** |
| Centralized DistilBERT vs. FedAvg + BiLSTM | 0.9863 | 0.9556 | +0.0307 | [0.0137, 0.0512] | 0.0000 | 7.1111 | 0.0077 | **Yes** |
| FedAvg + BiLSTM vs. Centralized BiLSTM | 0.9556 | 0.9659 | -0.0102 | [-0.0239, 0.0000] | 0.0920 | 1.3333 | 0.2482 | No |
| FedAvg + BiLSTM vs. Fed-DistilDP (Peak R1) | 0.9556 | 0.8464 | +0.1092 | [0.0614, 0.1536] | 0.0000 | 17.7963 | 0.0000 | **Yes** |
| FedAvg + BiLSTM vs. Fed-DistilDP (Final R10) | 0.9556 | 0.4846 | +0.4710 | [0.4061, 0.5324] | 0.0000 | 118.7911 | 0.0000 | **Yes** |


### Generalization and Sample Size Disclaimer
The UCI YouTube Spam Collection consists of 1,953 unique comments collected across five specific popular music videos from 2013-2015. While differences between contextual transformers, federated parameter exchange, and differentially private logit distillation are empirically measurable, broad generalization claims to modern platforms or evolving spam domains must be bounded by the finite corpus size and domain-specific vocabulary.
