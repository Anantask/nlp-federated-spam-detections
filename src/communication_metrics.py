"""
Communication Accounting Module for Federated Learning.
Accurately measures bidirectional network transmission payloads:
- Server-to-Client (Download): Global model parameters or auxiliary task data
- Client-to-Server (Upload): Local model weight updates or perturbed distillation logits
Calculates serialized byte sizes, cumulative bandwidth, and compression ratios.
"""

import io
import sys
from typing import Dict, Any, List, Optional
import numpy as np
import torch

class CommunicationTracker:
    """
    Tracks upload and download communication payloads across federated rounds.
    """
    def __init__(self):
        self.upload_bytes_by_round: Dict[int, int] = {}
        self.download_bytes_by_round: Dict[int, int] = {}
        self.round_history: List[Dict[str, Any]] = []

    @staticmethod
    def measure_model_payload(state_dict: Dict[str, torch.Tensor]) -> int:
        """
        Measures the exact wire serialized payload size (in bytes) of a PyTorch model state_dict.
        """
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        return buffer.getbuffer().nbytes

    @staticmethod
    def measure_logits_payload(logits: np.ndarray) -> int:
        """
        Measures the serialized payload size (in bytes) of transmitted prediction logits.
        """
        buffer = io.BytesIO()
        np.save(buffer, logits.astype(np.float32))
        return buffer.getbuffer().nbytes

    def record_round(
        self,
        round_idx: int,
        client_uploads: List[int],
        server_downloads: List[int],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Records communication metrics for a completed communication round.
        client_uploads: list of bytes transmitted by each participating client to server.
        server_downloads: list of bytes transmitted by server to each client.
        """
        total_upload = sum(client_uploads)
        total_download = sum(server_downloads)
        
        self.upload_bytes_by_round[round_idx] = total_upload
        self.download_bytes_by_round[round_idx] = total_download
        
        round_info = {
            "round": round_idx,
            "upload_bytes": total_upload,
            "download_bytes": total_download,
            "round_total_bytes": total_upload + total_download,
            "upload_kb": round(total_upload / 1024.0, 2),
            "download_kb": round(total_download / 1024.0, 2),
            "round_total_kb": round((total_upload + total_download) / 1024.0, 2),
        }
        if metadata:
            round_info.update(metadata)
            
        self.round_history.append(round_info)
        return round_info

    def get_summary(self) -> Dict[str, Any]:
        """
        Returns cumulative communication statistics.
        """
        total_upload = sum(self.upload_bytes_by_round.values())
        total_download = sum(self.download_bytes_by_round.values())
        total_bytes = total_upload + total_download
        
        return {
            "total_rounds": len(self.round_history),
            "cumulative_upload_bytes": total_upload,
            "cumulative_download_bytes": total_download,
            "cumulative_total_bytes": total_bytes,
            "cumulative_upload_mb": round(total_upload / (1024.0 * 1024.0), 3),
            "cumulative_download_mb": round(total_download / (1024.0 * 1024.0), 3),
            "cumulative_total_mb": round(total_bytes / (1024.0 * 1024.0), 3),
            "average_bytes_per_round": round(total_bytes / len(self.round_history), 2) if self.round_history else 0
        }
