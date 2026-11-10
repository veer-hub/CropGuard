"""
Crash-Resilient Atomic Checkpoint and Training Logger
CropGuard v3 - Power Outage & Fault Tolerance Engine

Guarantees:
1. Atomic writes via temporary files and os.replace (prevents corrupted .pt files if power cuts mid-save).
2. Append-only JSONL metric logging with immediate OS flush (os.fsync).
3. Seamless one-line state resumption from latest checkpoint.
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import numpy as np
except ImportError:
    np = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CheckpointManager")


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy scalars and arrays."""
    def default(self, obj):
        if np is not None:
            if isinstance(obj, (np.bool_, np.bool)):
                return bool(obj)
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        return super().default(obj)


class CheckpointManager:
    """Manages crash-proof checkpointing and logging for PyTorch training pipelines."""

    def __init__(self, checkpoint_dir: str = r"c:\CropGuard v3\checkpoints", experiment_name: str = "dinov2_lora"):
        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.checkpoint_dir / "training_history.jsonl"
        self.latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        self.best_path = self.checkpoint_dir / "best_checkpoint.pt"
        logger.info(f"Initialized CheckpointManager in: {self.checkpoint_dir}")

    def save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        val_metric: float = 0.0,
        is_best: bool = False,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Atomically saves checkpoint to disk.
        
        Uses .tmp write + fsync + os.replace to guarantee zero file corruption
        if power cuts during write.
        """
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metric": val_metric,
            "extra_state": extra_state or {},
        }
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()

        epoch_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        tmp_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt.tmp"

        # 1. Write to temporary file with physical fsync
        with open(tmp_path, "wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())

        # 2. Atomic rename to target
        os.replace(tmp_path, epoch_path)

        # 3. Update latest_checkpoint.pt atomically
        latest_tmp = self.checkpoint_dir / "latest_checkpoint.pt.tmp"
        with open(latest_tmp, "wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(latest_tmp, self.latest_path)

        # 4. Update best_checkpoint.pt if this is the best model so far
        if is_best:
            best_tmp = self.checkpoint_dir / "best_checkpoint.pt.tmp"
            with open(best_tmp, "wb") as f:
                torch.save(state, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(best_tmp, self.best_path)
            logger.info(f"[Power-Safe Checkpoint] Saved NEW BEST model (metric={val_metric:.4f}) at epoch {epoch}")

        logger.info(f"[Power-Safe Checkpoint] Epoch {epoch} committed atomically to physical disk.")
        return epoch_path

    def load_latest(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: torch.device = torch.device("cpu"),
    ) -> int:
        """Finds and resumes from the latest checkpoint if available.
        
        Returns starting epoch (0 if no checkpoint exists, or saved_epoch + 1).
        """
        if not self.latest_path.exists():
            logger.info("No prior checkpoint found. Starting training from scratch (Epoch 1).")
            return 1

        logger.info(f"Resuming from crash-safe checkpoint: {self.latest_path}")
        checkpoint = torch.load(self.latest_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        resumed_epoch = checkpoint["epoch"] + 1
        logger.info(f"Successfully restored weights & optimizer! Resuming at Epoch {resumed_epoch}.")
        return resumed_epoch

    def log_metrics(self, epoch: int, metrics: Dict[str, Any]):
        """Logs metrics to an append-only JSONL file with physical disk flush."""
        record = {
            "epoch": int(epoch),
            "metrics": metrics,
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, cls=NumpyEncoder) + "\n")
            f.flush()
            os.fsync(f.fileno())
