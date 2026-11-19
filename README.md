# Privacy-Preserving Federated NLP for Harmful-Content Detection using Fed-DistilDP

## 1. Project Overview

This project implements **Fed-DistilDP (Federated Knowledge Distillation with Local Differential Privacy and Adaptive Class-Imbalance Compensation)** for privacy-preserving harmful-content detection.

The system is evaluated using the real **UCI YouTube Spam Collection** dataset. It explores centralized machine learning, federated learning, knowledge distillation, and Local Differential Privacy.

Unlike conventional federated learning approaches that exchange model parameters, Fed-DistilDP communicates clipped and Gaussian-noise-perturbed prediction logits. This helps reduce communication overhead while providing a differential privacy mechanism.

### Main Objectives
- Detect spam and legitimate (ham) comments.
- Compare centralized and federated learning approaches.
- Implement knowledge distillation using DistilBERT.
- Apply Local Differential Privacy using L2 clipping and Gaussian noise.
- Study the effect of privacy noise on model performance.
- Evaluate communication cost, privacy, and predictive performance.

## 2. Dataset

**Dataset:** UCI YouTube Spam Collection  
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/380/youtube+spam+collection)

The dataset contains comments collected from five YouTube videos:
- Psy
- Katy Perry
- LMFAO
- Eminem
- Shakira

The original dataset contains 1,956 records. After removing 3 exact duplicate records, the cleaned dataset contains **1,953 unique comments**.

**Target labels:**
- 0 – Ham (legitimate comment)
- 1 – Spam

## 3. Proposed Methodology

The project compares multiple machine learning and deep learning approaches:

- TF-IDF with Logistic Regression
- TF-IDF with Linear SVM
- Bidirectional LSTM
- Centralized DistilBERT
- Federated Averaging (FedAvg)
- Proposed Fed-DistilDP framework

### Fed-DistilDP Workflow

1. Five clients train local DistilBERT teacher models using their respective data partitions.
2. Each client generates prediction logits on a public auxiliary text dataset.
3. The logits are L2-clipped and perturbed using Gaussian noise.
4. The central server aggregates the client predictions.
5. A student DistilBERT model learns from the aggregated predictions using KL-divergence-based knowledge distillation.
6. Rényi Differential Privacy (RDP) accounting is used to estimate the cumulative privacy budget.

## 4. Repository Structure

```text
NLP IA/
│
├── config/
│   ├── data.yaml
│   ├── federated.yaml
│   ├── privacy.yaml
│   └── training.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── [original dataset CSV files]
│
├── reports/
│   ├── data_quality_report.md
│   ├── experimental_results.md
│   ├── limitations.md
│   ├── methodology.md
│   └── privacy_analysis.md
│
├── results/
│   ├── comparison_tables/
│   ├── metrics/
│   └── plots/
│
├── scripts/
│   ├── download_data.py
│   ├── generate_visualizations.py
│   ├── inspect_raw_data.py
│   ├── prepare_splits.py
│   ├── run_ablations.py
│   ├── run_baselines.py
│   ├── run_distilbert.py
│   ├── run_federated.py
│   ├── run_fed_distildp.py
│   ├── run_statistical_tests.py
│   └── validate_data.py
│
├── src/
│   ├── classical_baselines.py
│   ├── client_partition.py
│   ├── communication_metrics.py
│   ├── data_loader.py
│   ├── data_split.py
│   ├── distilbert_model.py
│   ├── distillation.py
│   ├── federated_fedavg.py
│   ├── fed_distildp.py
│   ├── lstm_model.py
│   ├── metrics.py
│   ├── preprocessing.py
│   ├── privacy_accounting.py
│   ├── privacy_mechanism.py
│   └── statistical_analysis.py
│
├── tests/
├── requirements.txt
├── pytest.ini
└── README.md
```

## 5. Experimental Results

The project evaluates model performance using:
- Accuracy
- Precision
- Recall
- Macro-F1 Score
- ROC-AUC
- Communication Cost
- Differential Privacy Budget (epsilon and delta)

The experimental results, comparison tables, metric files, and generated plots are available in the `reports/` and `results/` directories.

The experiments also investigate:
- Centralized versus federated learning.
- Privacy-utility trade-offs under different Gaussian noise levels.
- The impact of focal loss.
- Model performance across communication rounds.
- Statistical comparisons and ablation studies.

Refer to `reports/experimental_results.md` and `results/` for the recorded experimental outputs.

## 6. Installation and Setup

Clone the repository:

```bash
git clone https://github.com/Anantask/nlp-federated-spam-detection.git
cd nlp-federated-spam-detection
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

**Requirements:** Python environment with the libraries specified in `requirements.txt`. A CUDA-compatible GPU is recommended for transformer training.

## 7. Running the Project

The project scripts can be executed from the repository root.

```bash
# Inspect and validate the raw dataset
python scripts/inspect_raw_data.py
python scripts/validate_data.py

# Prepare the dataset and train/test splits
python scripts/download_data.py
python scripts/prepare_splits.py

# Run baseline models
python scripts/run_baselines.py

# Train centralized DistilBERT
python scripts/run_distilbert.py

# Run federated learning
python scripts/run_federated.py

# Run the proposed Fed-DistilDP framework
python scripts/run_fed_distildp.py

# Run ablation experiments
python scripts/run_ablations.py

# Perform statistical tests
python scripts/run_statistical_tests.py

# Generate visualizations
python scripts/generate_visualizations.py
```

## 8. Testing

The project includes a test suite for checking data processing, model functionality, federated learning, and privacy mechanisms.

Run the tests using:

```bash
pytest tests/ -v
```

## 9. Limitations and Future Scope

The dataset is relatively small and contains comments from only five YouTube videos. The experiments also show performance degradation in later Fed-DistilDP communication rounds.

Future improvements could include:
- Evaluation on larger and more diverse datasets.
- Better auxiliary dataset selection.
- Adaptive privacy-noise scheduling.
- Early stopping and model-layer freezing.
- Parameter-efficient fine-tuning using LoRA.

## 10. Dataset Reference

Alberto, T. C., and Lochter, J. V. (2015). *YouTube Spam Collection*. UCI Machine Learning Repository.

https://doi.org/10.24432/C58885

---

**GitHub Repository:**  
https://github.com/Anantask/nlp-federated-spam-detections
