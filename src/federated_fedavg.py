"""
Federated Averaging (FedAvg) with BiLSTM:
Implements:
- Isolated FedAvgClient simulating private local training on video subsets
- FedAvgServer coordinating sample-size-weighted model parameter aggregation
- Full communication payload accounting (exact bytes for model broadcast and upload)
- Global model evaluation on held-out test data
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
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.lstm_model import BiLSTMClassifier, SimpleTextTokenizer, CommentTensorDataset
from src.metrics import compute_classification_metrics, Timer
from src.communication_metrics import CommunicationTracker
from src.client_partition import FederatedClientDataset


class FedAvgClient:
    """
    Simulates a federated client node holding private comment data.
    Trains locally and returns updated parameters.
    """
    def __init__(
        self,
        client_id: int,
        client_name: str,
        client_dataset: FederatedClientDataset,
        tokenizer: SimpleTextTokenizer,
        max_seq_len: int = 128,
        batch_size: int = 32,
        lr: float = 0.002,
        local_epochs: int = 3,
        device: Optional[torch.device] = None
    ):
        self.client_id = client_id
        self.client_name = client_name
        self.dataset = client_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.lr = lr
        self.local_epochs = local_epochs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Prepare private training tensors
        train_texts = self.dataset.train_df["processed_content"].tolist()
        train_labels = self.dataset.train_df["label"].tolist()
        X_train = self.tokenizer.transform(train_texts, max_len=self.max_seq_len)
        y_train = np.array(train_labels, dtype=int)
        
        self.train_loader = DataLoader(
            CommentTensorDataset(X_train, y_train),
            batch_size=min(self.batch_size, max(1, len(X_train))),
            shuffle=True
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_local(
        self,
        global_state_dict: Dict[str, torch.Tensor],
        model_template: BiLSTMClassifier
    ) -> Tuple[Dict[str, torch.Tensor], int, float]:
        """
        Loads global model weights, performs local training,
        and returns (local_state_dict, num_samples, average_loss).
        """
        local_model = copy.deepcopy(model_template).to(self.device)
        local_model.load_state_dict(global_state_dict)
        local_model.train()

        optimizer = torch.optim.Adam(local_model.parameters(), lr=self.lr)
        total_loss = 0.0
        total_steps = 0

        for _ in range(self.local_epochs):
            for seqs, labels in self.train_loader:
                seqs, labels = seqs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits = local_model(seqs)
                loss = self.criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_steps += 1

        avg_loss = total_loss / max(1, total_steps)
        # Move state dict to CPU before returning to emulate network transmission
        local_weights = {k: v.cpu().clone() for k, v in local_model.state_dict().items()}
        return local_weights, self.dataset.num_train_samples, avg_loss


class FedAvgServer:
    """
    Central server coordinating federated averaging rounds,
    parameter aggregation, communication accounting, and global evaluation.
    """
    def __init__(
        self,
        global_model: BiLSTMClassifier,
        tokenizer: SimpleTextTokenizer,
        clients: Dict[int, FedAvgClient],
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = global_model.to(self.device)
        self.tokenizer = tokenizer
        self.clients = clients
        self.comm_tracker = CommunicationTracker()
        self.round_history: List[Dict[str, Any]] = []

    def aggregate_weights(
        self,
        client_updates: List[Tuple[Dict[str, torch.Tensor], int]]
    ) -> Dict[str, torch.Tensor]:
        """
        Sample-size-weighted parameter averaging (McMahan et al., FedAvg).
        theta_global = sum( (n_k / N) * theta_k )
        """
        total_samples = sum(num_samples for _, num_samples in client_updates)
        if total_samples == 0:
            return self.global_model.state_dict()

        # Initialize accumulator with zeros
        first_weights, _ = client_updates[0]
        aggregated_weights = {k: torch.zeros_like(v) for k, v in first_weights.items()}

        for local_weights, num_samples in client_updates:
            weight_factor = num_samples / total_samples
            for k in aggregated_weights.keys():
                aggregated_weights[k] += local_weights[k].to(aggregated_weights[k].device) * weight_factor

        return aggregated_weights

    def evaluate_global(
        self,
        test_texts: List[str],
        test_labels: List[int],
        max_seq_len: int = 128
    ) -> Dict[str, Any]:
        """
        Evaluates current global model on held-out primary test set.
        """
        self.global_model.eval()
        X_test = self.tokenizer.transform(test_texts, max_len=max_seq_len)
        loader = DataLoader(
            CommentTensorDataset(X_test, np.zeros(len(test_texts))),
            batch_size=32,
            shuffle=False
        )

        all_preds = []
        all_probs = []
        with torch.no_grad():
            for seqs, _ in loader:
                seqs = seqs.to(self.device)
                logits = self.global_model(seqs)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                all_probs.append(probs)
                all_preds.append(preds)

        preds = np.concatenate(all_preds)
        probs = np.concatenate(all_probs)
        return compute_classification_metrics(test_labels, preds, probs[:, 1])

    def run_federated_rounds(
        self,
        num_rounds: int,
        test_texts: List[str],
        test_labels: List[int],
        max_seq_len: int = 128
    ) -> Dict[str, Any]:
        """
        Executes full federated learning procedure with communication tracking.
        """
        total_start_time = time.perf_counter()

        for round_idx in range(1, num_rounds + 1):
            round_start = time.perf_counter()
            # 1. Server broadcasts global model to all clients
            global_state = {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}
            model_bytes = CommunicationTracker.measure_model_payload(global_state)

            server_downloads = [model_bytes for _ in self.clients]
            client_uploads = []
            client_updates = []
            client_losses = []

            # 2. Clients train locally on their private data
            for cid, client in self.clients.items():
                local_weights, n_samples, loss = client.train_local(global_state, self.global_model)
                client_updates.append((local_weights, n_samples))
                upload_bytes = CommunicationTracker.measure_model_payload(local_weights)
                client_uploads.append(upload_bytes)
                client_losses.append(loss)

            # 3. Server aggregates client model parameters
            aggregated_state = self.aggregate_weights(client_updates)
            self.global_model.load_state_dict({k: v.to(self.device) for k, v in aggregated_state.items()})

            # 4. Measure communication
            round_comm = self.comm_tracker.record_round(
                round_idx=round_idx,
                client_uploads=client_uploads,
                server_downloads=server_downloads,
                metadata={"avg_client_loss": round(float(np.mean(client_losses)), 4)}
            )

            # 5. Evaluate global model on held-out test partition
            test_metrics = self.evaluate_global(test_texts, test_labels, max_seq_len=max_seq_len)
            round_duration = time.perf_counter() - round_start

            round_summary = {
                "round": round_idx,
                "round_time_sec": round(round_duration, 2),
                "avg_client_loss": round(float(np.mean(client_losses)), 4),
                "accuracy": test_metrics["accuracy"],
                "macro_f1": test_metrics["macro_f1"],
                "spam_f1": test_metrics["spam_f1"],
                "roc_auc": test_metrics["roc_auc"],
                "communication_kb": round_comm["round_total_kb"]
            }
            self.round_history.append(round_summary)
            print(f"Round {round_idx:02d}/{num_rounds:02d} | Avg Loss: {round_summary['avg_client_loss']:.4f} | Test Acc: {test_metrics['accuracy']:.4f} | Test Macro-F1: {test_metrics['macro_f1']:.4f} | Comm: {round_comm['round_total_kb']} KB")

        total_duration = time.perf_counter() - total_start_time
        final_test_metrics = self.evaluate_global(test_texts, test_labels, max_seq_len=max_seq_len)
        comm_summary = self.comm_tracker.get_summary()

        return {
            "model_name": "Federated BiLSTM (FedAvg)",
            "num_clients": len(self.clients),
            "num_rounds": num_rounds,
            "total_time_sec": round(total_duration, 2),
            "final_metrics": final_test_metrics,
            "communication_summary": comm_summary,
            "round_history": self.round_history
        }
