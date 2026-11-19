"""
Proposed Method: Fed-DistilDP
Federated Knowledge Distillation with Local Differential Privacy
and Adaptive Class-Imbalance Compensation.

Components:
1. Local DistilBERT teachers trained with class-weighted Focal Loss (Adaptive Imbalance Compensation)
2. Client-side L2 clipping and calibrated Gaussian noise injection (Local Differential Privacy)
3. Server-side sample-weighted logit aggregation over isolated auxiliary transfer set
4. Global student model distillation using temperature-softened KL-divergence loss
5. Formal Rényi Differential Privacy (RDP) accounting and bidirectional wire communication tracking
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

from src.distilbert_model import DistilBERTSequenceClassifier, DistilBertTextDataset
from src.distillation import FocalLoss, DistillationLoss
from src.privacy_mechanism import GaussianLDPMechanism
from src.privacy_accounting import RDPPrivacyAccountant
from src.communication_metrics import CommunicationTracker
from src.client_partition import FederatedClientDataset
from src.metrics import compute_classification_metrics, Timer


class FedDistilDPClient:
    """
    Simulates a privacy-preserving federated client node under Fed-DistilDP.
    Maintains isolated private comment training data.
    Trains a local teacher model using class-weighted Focal Loss.
    Applies LDP mechanism (L2 clipping + Gaussian noise) on auxiliary logits prior to transmission.
    """
    def __init__(
        self,
        client_id: int,
        client_name: str,
        client_dataset: FederatedClientDataset,
        max_seq_len: int = 128,
        batch_size: int = 16,
        lr: float = 3e-5,
        focal_gamma: float = 2.0,
        device: Optional[torch.device] = None,
        freeze_backbone: bool = True
    ):
        self.client_id = client_id
        self.client_name = client_name
        self.dataset = client_dataset
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.lr = lr
        self.focal_gamma = focal_gamma
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.freeze_backbone = freeze_backbone

        # Calculate class weights alpha from local private training set
        counts = self.dataset.class_counts
        total_samples = self.dataset.num_train_samples
        # alpha for class 1 = ratio of class 0, and vice versa
        if counts[1] > 0 and counts[0] > 0:
            alpha_0 = float(counts[1] / total_samples)
            alpha_1 = float(counts[0] / total_samples)
            self.class_weights = [alpha_0, alpha_1]
        else:
            self.class_weights = [0.5, 0.5]

        # Initialize local teacher model
        self.teacher = DistilBERTSequenceClassifier(
            model_name="distilbert-base-uncased",
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            lr=lr,
            freeze_backbone=freeze_backbone,
            device=str(self.device)
        )

    def train_teacher(self, local_epochs: int = 2) -> Dict[str, Any]:
        """
        Trains local teacher on private training partition using Focal Loss.
        """
        train_texts = self.dataset.train_df["processed_content"].tolist()
        train_labels = self.dataset.train_df["label"].tolist()

        train_enc = self.teacher.tokenize_texts(train_texts)
        train_loader = DataLoader(
            DistilBertTextDataset(train_enc, train_labels),
            batch_size=self.batch_size,
            shuffle=True
        )

        focal_criterion = FocalLoss(
            alpha=self.class_weights,
            gamma=self.focal_gamma
        )
        optimizer = torch.optim.AdamW(
            [p for p in self.teacher.model.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=0.01
        )

        self.teacher.model.train()
        total_loss = 0.0
        total_steps = 0

        for _ in range(local_epochs):
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.teacher.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = focal_criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_steps += 1

        avg_loss = total_loss / max(1, total_steps)
        return {
            "client_id": self.client_id,
            "client_name": self.client_name,
            "train_loss": round(avg_loss, 4),
            "class_weights": self.class_weights
        }

    def generate_private_auxiliary_logits(
        self,
        auxiliary_texts: List[str],
        ldp_mechanism: GaussianLDPMechanism
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Computes prediction logits on auxiliary transfer set, applies client-side
        L2 norm clipping bound C and Gaussian noise, and returns perturbed logits.
        """
        self.teacher.model.eval()
        enc = self.teacher.tokenize_texts(auxiliary_texts)
        loader = DataLoader(
            DistilBertTextDataset(enc),
            batch_size=self.batch_size,
            shuffle=False
        )
        all_logits = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.teacher.model(input_ids=input_ids, attention_mask=attention_mask)
                all_logits.append(outputs.logits.cpu().numpy())

        raw_logits = np.concatenate(all_logits, axis=0)

        # Apply client-side LDP perturbation
        perturbed_logits, diagnostics = ldp_mechanism.perturb(raw_logits)
        diagnostics["client_id"] = self.client_id
        diagnostics["client_name"] = self.client_name

        return perturbed_logits, diagnostics


class FedDistilDPServer:
    """
    Central coordinator for Fed-DistilDP:
    - Maintains the isolated public auxiliary dataset
    - Collects differentially private perturbed logits from clients
    - Aggregates logits using sample-weighted averaging
    - Distills the global student model
    - Tracks RDP privacy composition and bidirectional communication payloads
    """
    def __init__(
        self,
        auxiliary_texts: List[str],
        clients: Dict[int, FedDistilDPClient],
        ldp_mechanism: GaussianLDPMechanism,
        privacy_accountant: RDPPrivacyAccountant,
        distillation_temperature: float = 2.0,
        student_lr: float = 5e-5,
        max_seq_len: int = 128,
        device: Optional[torch.device] = None,
        freeze_student_backbone: bool = False
    ):
        self.auxiliary_texts = auxiliary_texts
        self.clients = clients
        self.ldp_mech = ldp_mechanism
        self.accountant = privacy_accountant
        self.temperature = distillation_temperature
        self.student_lr = student_lr
        self.max_seq_len = max_seq_len
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.comm_tracker = CommunicationTracker()
        self.round_history: List[Dict[str, Any]] = []

        # Global Student Model
        self.student = DistilBERTSequenceClassifier(
            model_name="distilbert-base-uncased",
            max_seq_len=max_seq_len,
            batch_size=16,
            lr=student_lr,
            freeze_backbone=freeze_student_backbone,
            device=str(self.device)
        )
        self.distill_criterion = DistillationLoss(temperature=distillation_temperature, mode="kl_div")

    def aggregate_logits(
        self,
        client_logits_list: List[Tuple[np.ndarray, int]]
    ) -> np.ndarray:
        """
        Sample-weighted aggregation of client perturbed logits:
        z_agg = sum( (n_k / N) * z_k_perturbed )
        """
        total_samples = sum(n for _, n in client_logits_list)
        aggregated = np.zeros_like(client_logits_list[0][0], dtype=np.float32)

        for logits, n in client_logits_list:
            weight = n / total_samples
            aggregated += logits * weight

        return aggregated

    def train_student_step(
        self,
        aggregated_logits: np.ndarray,
        distill_epochs: int = 2
    ) -> float:
        """
        Distills global student model on auxiliary dataset using aggregated soft targets.
        """
        self.student.model.train()
        enc = self.student.tokenize_texts(self.auxiliary_texts)
        targets_tensor = torch.tensor(aggregated_logits, dtype=torch.float32)

        class DistillDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, targets):
                self.enc = encodings
                self.tgt = targets
            def __len__(self):
                return len(self.tgt)
            def __getitem__(self, i):
                return {
                    "input_ids": self.enc["input_ids"][i],
                    "attention_mask": self.enc["attention_mask"][i],
                    "targets": self.tgt[i]
                }

        loader = DataLoader(
            DistillDataset(enc, targets_tensor),
            batch_size=16,
            shuffle=True
        )

        optimizer = torch.optim.AdamW(
            [p for p in self.student.model.parameters() if p.requires_grad],
            lr=self.student_lr,
            weight_decay=0.01
        )

        total_loss = 0.0
        total_batches = 0

        for _ in range(distill_epochs):
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                teacher_logits = batch["targets"].to(self.device)

                optimizer.zero_grad()
                outputs = self.student.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.distill_criterion(outputs.logits, teacher_logits)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_batches += 1

        return total_loss / max(1, total_batches)

    def evaluate_student(
        self,
        test_texts: List[str],
        test_labels: List[int]
    ) -> Dict[str, Any]:
        """Evaluates distilled global student on held-out primary test set."""
        return self.student.evaluate(test_texts, test_labels)

    def run_distildp_rounds(
        self,
        num_rounds: int,
        local_epochs: int,
        test_texts: List[str],
        test_labels: List[int],
        distill_epochs: int = 2
    ) -> Dict[str, Any]:
        """
        Executes end-to-end Fed-DistilDP multi-round training.
        """
        total_start = time.perf_counter()

        # Step 1: Local teacher training on private client data
        print(f"\n[INFO] Phase 1: Local Client Teacher Training with Focal Loss (gamma={self.clients[0].focal_gamma})...")
        client_train_summaries = []
        for cid, client in self.clients.items():
            summary = client.train_teacher(local_epochs=local_epochs)
            client_train_summaries.append(summary)
            print(f"  Client {cid} ({client.client_name}) Teacher trained | Loss: {summary['train_loss']} | Class Weights: {summary['class_weights']}")

        print(f"\n[INFO] Phase 2: Knowledge Distillation across {num_rounds} rounds (Auxiliary Size={len(self.auxiliary_texts)}, Noise sigma={self.ldp_mech.sigma}, Clip C={self.ldp_mech.C})...")

        best_round_f1 = -1.0
        best_round_summary = None
        best_round_idx = 1
        best_student_weights = None

        for r in range(1, num_rounds + 1):
            round_start = time.perf_counter()

            # Server broadcasts request / auxiliary task data
            # Transmitting N_aux texts or task ID: measured as token length bytes
            aux_payload_bytes = sum(len(t.encode("utf-8")) for t in self.auxiliary_texts)
            server_downloads = [aux_payload_bytes for _ in self.clients]

            client_uploads = []
            client_logits_list = []
            client_diagnostics = []

            # Clients generate perturbed logits locally
            for cid, client in self.clients.items():
                perturbed_logits, diag = client.generate_private_auxiliary_logits(
                    self.auxiliary_texts, self.ldp_mech
                )
                client_logits_list.append((perturbed_logits, client.dataset.num_train_samples))
                upload_bytes = CommunicationTracker.measure_logits_payload(perturbed_logits)
                client_uploads.append(upload_bytes)
                client_diagnostics.append(diag)

            # Server aggregates perturbed logits
            agg_logits = self.aggregate_logits(client_logits_list)

            # Server distills student model using aggregated soft targets
            distill_loss = self.train_student_step(agg_logits, distill_epochs=distill_epochs)

            # Record communication payload for this round
            round_comm = self.comm_tracker.record_round(
                round_idx=r,
                client_uploads=client_uploads,
                server_downloads=server_downloads,
                metadata={"distill_loss": round(distill_loss, 4)}
            )

            # Evaluate global student on held-out primary test set
            eval_res = self.evaluate_student(test_texts, test_labels)
            tm = eval_res["metrics"]
            round_duration = time.perf_counter() - round_start

            # Calculate cumulative DP epsilon for r releases
            privacy_rep = self.accountant.generate_privacy_report(
                sigma=self.ldp_mech.sigma,
                rounds=r
            )
            eps_val = privacy_rep["epsilon"]
            eps_str = f"eps={eps_val:.2f}" if isinstance(eps_val, float) else str(eps_val)

            round_summary = {
                "round": r,
                "round_time_sec": round(round_duration, 2),
                "distill_loss": round(distill_loss, 4),
                "accuracy": tm["accuracy"],
                "macro_f1": tm["macro_f1"],
                "spam_f1": tm["spam_f1"],
                "roc_auc": tm["roc_auc"],
                "epsilon": eps_val,
                "communication_kb": round_comm["round_total_kb"]
            }
            self.round_history.append(round_summary)

            if tm["macro_f1"] > best_round_f1:
                best_round_f1 = tm["macro_f1"]
                best_round_summary = copy.deepcopy(round_summary)
                best_round_idx = r
                best_student_weights = copy.deepcopy(self.student.model.state_dict())

            print(f"Fed-DistilDP Round {r:02d}/{num_rounds:02d} | Distill Loss: {distill_loss:.4f} | Test Acc: {tm['accuracy']:.4f} | Macro-F1: {tm['macro_f1']:.4f} | {eps_str} | Comm: {round_comm['round_total_kb']} KB")

        total_time = time.perf_counter() - total_start
        final_test_eval = self.evaluate_student(test_texts, test_labels)
        comm_summary = self.comm_tracker.get_summary()
        final_privacy = self.accountant.generate_privacy_report(
            sigma=self.ldp_mech.sigma,
            rounds=num_rounds
        )

        return {
            "model_name": "Fed-DistilDP (Proposed)",
            "num_clients": len(self.clients),
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "distillation_epochs": distill_epochs,
            "focal_gamma": self.clients[0].focal_gamma,
            "noise_multiplier_sigma": self.ldp_mech.sigma,
            "clipping_bound_C": self.ldp_mech.C,
            "temperature": self.temperature,
            "total_time_sec": round(total_time, 2),
            "final_metrics": final_test_eval["metrics"],
            "best_round": best_round_idx,
            "best_round_metrics": best_round_summary,
            "privacy_guarantee": final_privacy,
            "communication_summary": comm_summary,
            "client_training_summaries": client_train_summaries,
            "round_history": self.round_history
        }

