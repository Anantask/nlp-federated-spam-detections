"""
Transformer Baseline: DistilBERT Sequence Classifier.
Implements:
- Centralized fine-tuning of distilbert-base-uncased for spam detection
- Tokenization with padding/truncation
- Validation checkpointing and early stopping
- Modular architecture reusable for Federated Transformer (FedNLP baseline) and Fed-DistilDP
"""

import os
import sys
import copy
import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AutoConfig

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.metrics import compute_classification_metrics, Timer


class DistilBertTextDataset(Dataset):
    """PyTorch Dataset wrapping tokenized DistilBERT encodings and targets."""
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: Optional[List[int]] = None):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


class DistilBERTSequenceClassifier:
    """
    DistilBERT-based binary sequence classification pipeline.
    """
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_seq_len: int = 128,
        batch_size: int = 16,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        epochs: int = 4,
        patience: int = 3,
        freeze_backbone: bool = False,
        device: Optional[str] = None,
        random_seed: int = 42
    ):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.freeze_backbone = freeze_backbone
        self.random_seed = random_seed

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        ).to(self.device)

        if freeze_backbone:
            # Freeze transformer encoder layers, train only classification head
            for param in self.model.distilbert.parameters():
                param.requires_grad = False

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "val_macro_f1": []
        }
        self.best_model_weights = None
        self.train_time = 0.0

    def tokenize_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt"
        )

    def fit(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int]
    ) -> Dict[str, Any]:
        with Timer() as t:
            train_enc = self.tokenize_texts(train_texts)
            val_enc = self.tokenize_texts(val_texts)

            train_loader = DataLoader(
                DistilBertTextDataset(train_enc, train_labels),
                batch_size=self.batch_size,
                shuffle=True
            )
            val_loader = DataLoader(
                DistilBertTextDataset(val_enc, val_labels),
                batch_size=self.batch_size,
                shuffle=False
            )

            # AdamW optimizer with weight decay on non-bias/LayerNorm params
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)

            best_val_f1 = -1.0
            patience_counter = 0

            for epoch in range(1, self.epochs + 1):
                self.model.train()
                epoch_train_loss = 0.0
                for batch in train_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss += loss.item() * len(labels)

                epoch_train_loss /= len(train_loader.dataset)

                # Validation phase
                self.model.eval()
                epoch_val_loss = 0.0
                val_preds, val_targets = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)
                        labels = batch["labels"].to(self.device)

                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        epoch_val_loss += outputs.loss.item() * len(labels)
                        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                        val_preds.extend(preds)
                        val_targets.extend(labels.cpu().numpy())

                epoch_val_loss /= len(val_loader.dataset)
                val_metrics = compute_classification_metrics(val_targets, val_preds)
                val_f1 = val_metrics["macro_f1"]

                self.history["train_loss"].append(round(epoch_train_loss, 4))
                self.history["val_loss"].append(round(epoch_val_loss, 4))
                self.history["val_macro_f1"].append(round(val_f1, 4))

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    self.best_model_weights = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

            # Restore best checkpoint
            if self.best_model_weights is not None:
                self.model.load_state_dict(self.best_model_weights)

        self.train_time = t.elapsed
        return {
            "train_time_sec": round(getattr(self, "train_time", 0.0), 4),
            "epochs_trained": len(self.history["train_loss"]),
            "best_val_macro_f1": round(best_val_f1, 4),
            "history": self.history
        }

    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        encodings = self.tokenize_texts(texts)
        loader = DataLoader(
            DistilBertTextDataset(encodings),
            batch_size=self.batch_size,
            shuffle=False
        )
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
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

        # Calculate model size in MB
        param_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        model_size_mb = param_bytes / (1024.0 * 1024.0)

        return {
            "model_name": "Centralized DistilBERT",
            "model_type": "distilbert",
            "train_time_sec": round(getattr(self, "train_time", 0.0), 4),
            "inference_time_sec": round(inference_time, 4),
            "inference_per_sample_ms": round(per_sample_ms, 3),
            "model_size_mb": round(model_size_mb, 2),
            "device": str(self.device),
            "metrics": metrics,
            "training_history": self.history
        }
