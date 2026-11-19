"""
Federated Client Partitioning Module.
Implements:
1. Configuration A (Source-Based Partitioning): Each video source forms a client.
2. Configuration B (Controlled Non-IID Label Skew): Dirichlet-based label skew
   partitioning across 5 clients.
Provides client data isolation and summary statistics for reporting.
"""

import os
import sys
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_CLIENTS = ["Psy", "KatyPerry", "LMFAO", "Eminem", "Shakira"]


class FederatedClientDataset:
    """
    Encapsulates private client data partitions.
    Ensures the central server only interacts with models/logits, not raw comments.
    """
    def __init__(
        self,
        client_id: int,
        client_name: str,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None
    ):
        self.client_id = client_id
        self.client_name = client_name
        self._train_df = train_df.copy().reset_index(drop=True)
        self._val_df = val_df.copy().reset_index(drop=True) if val_df is not None else None

    @property
    def train_df(self) -> pd.DataFrame:
        return self._train_df

    @property
    def val_df(self) -> Optional[pd.DataFrame]:
        return self._val_df

    @property
    def num_train_samples(self) -> int:
        return len(self._train_df)

    @property
    def num_val_samples(self) -> int:
        return len(self._val_df) if self._val_df is not None else 0

    @property
    def class_counts(self) -> Dict[int, int]:
        counts = self._train_df["label"].value_counts().to_dict()
        return {0: counts.get(0, 0), 1: counts.get(1, 0)}

    @property
    def spam_ratio(self) -> float:
        counts = self.class_counts
        total = counts[0] + counts[1]
        return counts[1] / total if total > 0 else 0.0

    def get_summary(self) -> Dict[str, Any]:
        counts = self.class_counts
        return {
            "client_id": self.client_id,
            "client_name": self.client_name,
            "num_train_samples": self.num_train_samples,
            "num_val_samples": self.num_val_samples,
            "spam_count": counts[1],
            "ham_count": counts[0],
            "spam_ratio": round(self.spam_ratio, 4)
        }


def partition_clients_source_based(
    df_splits: pd.DataFrame,
    client_names: Optional[List[str]] = None
) -> Dict[int, FederatedClientDataset]:
    """
    Configuration A — Natural source-based partitioning.
    Each of the 5 YouTube video sources corresponds to an isolated federated client.
    """
    if client_names is None:
        client_names = DEFAULT_CLIENTS

    clients = {}
    for idx, name in enumerate(client_names):
        # Extract private training and local validation partitions for this source
        client_train = df_splits[
            (df_splits["video_source"] == name) & 
            (df_splits["split"] == "train_private")
        ]
        client_val = df_splits[
            (df_splits["video_source"] == name) & 
            (df_splits["split"] == "val")
        ]
        
        clients[idx] = FederatedClientDataset(
            client_id=idx,
            client_name=name,
            train_df=client_train,
            val_df=client_val
        )
        
    return clients


def partition_clients_dirichlet_non_iid(
    df_splits: pd.DataFrame,
    num_clients: int = 5,
    alpha: float = 0.5,
    random_seed: int = 42
) -> Dict[int, FederatedClientDataset]:
    """
    Configuration B — Controlled non-IID label skew partitioning.
    Applies Dirichlet distribution Dir(alpha) to partition the pooled private
    training data across clients, creating realistic heterogeneous class distributions.
    Validation data is also partitioned consistently.
    """
    np.random.seed(random_seed)
    train_pool = df_splits[df_splits["split"] == "train_private"].copy().reset_index(drop=True)
    val_pool = df_splits[df_splits["split"] == "val"].copy().reset_index(drop=True)
    
    num_classes = 2 # Binary: Ham (0), Spam (1)
    client_train_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}
    
    # Partition training pool using Dirichlet label proportions
    for c in range(num_classes):
        class_indices = np.where(train_pool["label"] == c)[0]
        np.random.shuffle(class_indices)
        
        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        # Normalize and compute split cutoffs
        proportions = proportions / proportions.sum()
        split_sizes = (proportions * len(class_indices)).astype(int)
        
        # Ensure all samples are assigned by adjusting the last client
        split_sizes[-1] = len(class_indices) - split_sizes[:-1].sum()
        
        current_idx = 0
        for client_id, size in enumerate(split_sizes):
            assigned = class_indices[current_idx : current_idx + size]
            client_train_indices[client_id].extend(assigned)
            current_idx += size
            
    # Partition validation pool proportionally
    val_indices = np.arange(len(val_pool))
    np.random.shuffle(val_indices)
    val_splits = np.array_split(val_indices, num_clients)
    
    clients = {}
    for i in range(num_clients):
        c_train = train_pool.iloc[client_train_indices[i]]
        c_val = val_pool.iloc[val_splits[i]]
        client_name = f"NonIID_Client_{i+1}"
        clients[i] = FederatedClientDataset(
            client_id=i,
            client_name=client_name,
            train_df=c_train,
            val_df=c_val
        )
        
    return clients


def generate_client_partition_report(
    clients_a: Dict[int, FederatedClientDataset],
    clients_b: Dict[int, FederatedClientDataset]
) -> Dict[str, Any]:
    """
    Summarizes client distribution metrics for both Configuration A and B.
    """
    return {
        "config_a_source_based": [c.get_summary() for c in clients_a.values()],
        "config_b_controlled_non_iid": [c.get_summary() for c in clients_b.values()]
    }
