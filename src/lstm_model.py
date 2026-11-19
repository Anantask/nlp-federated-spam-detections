"""
Recurrent Neural Network Baseline: Bidirectional LSTM Classifier.
Includes:
- Text tokenizer and vocabulary mapping fitted strictly on training data
- PyTorch BiLSTM neural architecture with embedding and dropout
- Centralized training pipeline with early stopping and checkpoint tracking
- Reusable for Federated Averaging (FedAvg) client training
"""

import os
import sys
import re
import time
import copy
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.metrics import compute_classification_metrics, Timer


class SimpleTextTokenizer:
    """
    Lightweight whitespace and punctuation tokenizer.
    Vocabulary is learned exclusively from training text.
    """
    def __init__(self, vocab_size: int = 5000, min_freq: int = 2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.word2idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx2word = {0: self.pad_token, 1: self.unk_token}

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase and split on non-alphanumeric tokens, preserving word tokens
        return re.findall(r'\b\w+\b', str(text).lower())

    def fit(self, texts: List[str]) -> "SimpleTextTokenizer":
        counter = Counter()
        for t in texts:
            counter.update(self._tokenize(t))
            
        most_common = counter.most_common(self.vocab_size - 2)
        idx = 2
        for word, count in most_common:
            if count >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        return self

    def transform(self, texts: List[str], max_len: int = 128) -> np.ndarray:
        padded = np.zeros((len(texts), max_len), dtype=np.int64)
        unk_idx = self.word2idx[self.unk_token]
        
        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            indices = [self.word2idx.get(w, unk_idx) for w in tokens[:max_len]]
            if len(indices) > 0:
                padded[i, :len(indices)] = indices
        return padded


class CommentTensorDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM Classifier for Spam Detection.
    """
    def __init__(
        self,
        vocab_size: int = 5000,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        # Global max pooling over sequence length
        pooled, _ = torch.max(lstm_out, dim=1)
        logits = self.fc(pooled)
        return logits


class CentralizedLSTMPipeline:
    """
    Handles end-to-end training and evaluation of Centralized LSTM.
    """
    def __init__(
        self,
        vocab_size: int = 5000,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.3,
        max_seq_len: int = 128,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        epochs: int = 25,
        patience: int = 5,
        device: Optional[str] = None,
        random_seed: int = 42
    ):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.lr = lr
        self.weight_decay = weight_decay
        self.random_seed = random_seed
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = SimpleTextTokenizer(vocab_size=vocab_size)
        self.model = BiLSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "val_macro_f1": []
        }
        self.best_model_weights = None

    def fit(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int]
    ) -> Dict[str, Any]:
        with Timer() as t:
            # 1. Fit tokenizer exclusively on training texts
            self.tokenizer.fit(train_texts)
            X_train = self.tokenizer.transform(train_texts, max_len=self.max_seq_len)
            y_train = np.array(train_labels, dtype=int)
            
            X_val = self.tokenizer.transform(val_texts, max_len=self.max_seq_len)
            y_val = np.array(val_labels, dtype=int)

            train_loader = DataLoader(
                CommentTensorDataset(X_train, y_train),
                batch_size=self.batch_size,
                shuffle=True
            )
            val_loader = DataLoader(
                CommentTensorDataset(X_val, y_val),
                batch_size=self.batch_size,
                shuffle=False
            )

            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

            best_val_f1 = -1.0
            patience_counter = 0

            for epoch in range(1, self.epochs + 1):
                # Training phase
                self.model.train()
                epoch_train_loss = 0.0
                for seqs, labels in train_loader:
                    seqs, labels = seqs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    logits = self.model(seqs)
                    loss = self.criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss += loss.item() * len(labels)

                epoch_train_loss /= len(train_loader.dataset)

                # Validation phase
                self.model.eval()
                epoch_val_loss = 0.0
                val_preds, val_targets = [], []
                with torch.no_grad():
                    for seqs, labels in val_loader:
                        seqs, labels = seqs.to(self.device), labels.to(self.device)
                        logits = self.model(seqs)
                        loss = self.criterion(logits, labels)
                        epoch_val_loss += loss.item() * len(labels)
                        preds = torch.argmax(logits, dim=1).cpu().numpy()
                        val_preds.extend(preds)
                        val_targets.extend(labels.cpu().numpy())

                epoch_val_loss /= len(val_loader.dataset)
                val_metrics = compute_classification_metrics(val_targets, val_preds)
                val_f1 = val_metrics["macro_f1"]

                self.history["train_loss"].append(round(epoch_train_loss, 4))
                self.history["val_loss"].append(round(epoch_val_loss, 4))
                self.history["val_macro_f1"].append(round(val_f1, 4))

                # Checkpoint best validation model
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    self.best_model_weights = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

            # Restore best weights
            if self.best_model_weights is not None:
                self.model.load_state_dict(self.best_model_weights)

        self.train_time = t.elapsed
        return {
            "train_time_sec": round(self.train_time, 4),
            "epochs_trained": len(self.history["train_loss"]),
            "best_val_macro_f1": round(best_val_f1, 4),
            "history": self.history
        }

    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        X = self.tokenizer.transform(texts, max_len=self.max_seq_len)
        loader = DataLoader(
            CommentTensorDataset(X, np.zeros(len(texts))),
            batch_size=self.batch_size,
            shuffle=False
        )
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for seqs, _ in loader:
                seqs = seqs.to(self.device)
                logits = self.model(seqs)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                all_probs.append(probs)
                all_preds.append(preds)

        return np.concatenate(all_preds), np.concatenate(all_probs)

    def evaluate(self, test_texts: List[str], test_labels: List[int]) -> Dict[str, Any]:
        with Timer() as t:
            preds, probs = self.predict(test_texts)
        inference_time = t.elapsed
        per_sample_ms = (inference_time / len(test_texts)) * 1000.0 if len(test_texts) > 0 else 0.0

        metrics = compute_classification_metrics(test_labels, preds, probs[:, 1])
        
        # Calculate model size in KB
        param_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        
        return {
            "model_name": "Centralized BiLSTM",
            "model_type": "lstm",
            "train_time_sec": round(self.train_time, 4),
            "inference_time_sec": round(inference_time, 4),
            "inference_per_sample_ms": round(per_sample_ms, 3),
            "model_size_kb": round(param_bytes / 1024.0, 2),
            "device": str(self.device),
            "metrics": metrics,
            "training_history": self.history
        }
