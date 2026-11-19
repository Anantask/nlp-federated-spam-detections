# Baseline Evaluation on YouTube Spam Collection (Primary Test Set)

| Model | Accuracy | Macro-F1 | Weighted-F1 | Spam Precision | Spam Recall | Spam F1 | ROC-AUC | Train Time (s) | Inference (ms/sample) | Size (KB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| TF-IDF + Logistic Regression | 0.9761 | 0.9761 | 0.9761 | 0.9865 | 0.9669 | 0.9766 | 0.9921 | 0.116 | 0.032 | 132.9 |
| TF-IDF + Linear Svm | 0.9590 | 0.9589 | 0.9590 | 0.9427 | 0.9801 | 0.9610 | 0.9927 | 0.1356 | 0.084 | 179.75 |
| Centralized BiLSTM | 0.9659 | 0.9658 | 0.9658 | 0.9490 | 0.9868 | 0.9675 | 0.9944 | 11.3354 | 0.161 | 2920.76 |
