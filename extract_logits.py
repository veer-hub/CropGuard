"""
Phase 4 - Step 4.1: Logit Extraction Engine
CropGuard v3 - Calibration, Uncertainty Quantification & Explainability

Executes deterministic forward inference using the frozen `best_checkpoint.pt` model
on the validation set (1,546 images) and held-out test set (1,588 images).
Saves unnormalized logit matrices, ground-truth targets, class mappings, and image paths
into an atomic disk cache (`checkpoints/dinov2_lora/logits_cache.pt`).
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

# Ensure workspace root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_dinov2_lora import (
    DinoV2LoRAPathologyModel,
    PlantDocCleanDataset,
    get_val_transforms,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("LogitExtractor")


@torch.no_grad()
def extract_split_logits(
    model: nn.Module,
    dataset: PlantDocCleanDataset,
    batch_size: int = 32,
    num_workers: int = 0,
    device: Optional[torch.device] = None,
    use_amp: bool = True,
) -> Dict[str, Any]:
    """
    Runs forward pass over a dataset split and collects unnormalized float32 logits,
    ground-truth targets, and sample file paths in deterministic order.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    sample_paths: List[str] = [str(p) for p, _ in dataset.samples]

    start_time = time.time()
    num_batches = len(loader)

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(use_amp and device.type == "cuda"),
        ):
            logits, _ = model(images, return_attention=False)

        # Cast to float32 for high-precision numerical optimization in temperature scaling
        all_logits.append(logits.detach().float().cpu())
        all_targets.append(targets.detach().long().cpu())

        if (step + 1) % 20 == 0 or (step + 1) == num_batches:
            elapsed = time.time() - start_time
            logger.info(
                f"Split [{dataset.split_dir.name}] Batch [{step + 1:03d}/{num_batches:03d}] "
                f"| Elapsed: {elapsed:.1f}s"
            )

    logits_tensor = torch.cat(all_logits, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)

    preds = torch.argmax(logits_tensor, dim=-1).numpy()
    targets_np = targets_tensor.numpy()
    acc = float(accuracy_score(targets_np, preds))
    macro_f1 = float(f1_score(targets_np, preds, average="macro", zero_division=0))

    logger.info(
        f"Split [{dataset.split_dir.name}] Complete: N={len(targets_tensor)} | "
        f"Accuracy={acc * 100:.2f}% | Macro-F1={macro_f1:.4f}"
    )

    return {
        "logits": logits_tensor,
        "targets": targets_tensor,
        "paths": sample_paths,
        "accuracy": acc,
        "macro_f1": macro_f1,
    }


def run_logit_extraction(
    data_dir: Path,
    checkpoint_path: Path,
    output_cache_path: Path,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 47,
) -> Dict[str, Any]:
    """Loads best_checkpoint.pt, extracts val and test logits, and saves atomically."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    val_transform = get_val_transforms(image_size=224)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    # Build canonical class mapping from train split (or checkpoint)
    ckpt = torch.load(checkpoint_path, map_location=device)
    extra_state = ckpt.get("extra_state", {})
    saved_classes = extra_state.get("classes", None)

    if saved_classes is not None:
        class_to_idx = {c: i for i, c in enumerate(saved_classes)}
    else:
        train_ds = PlantDocCleanDataset(train_dir, transform=val_transform)
        class_to_idx = train_ds.class_to_idx

    val_dataset = PlantDocCleanDataset(val_dir, transform=val_transform, class_to_idx=class_to_idx)
    test_dataset = PlantDocCleanDataset(test_dir, transform=val_transform, class_to_idx=class_to_idx)

    num_classes = len(val_dataset.classes)
    model = DinoV2LoRAPathologyModel(
        num_classes=num_classes,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        pooling_heads=8,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    ckpt_epoch = int(ckpt.get("epoch", -1))
    logger.info(f"Loaded best_checkpoint.pt from epoch {ckpt_epoch}")

    val_res = extract_split_logits(
        model=model,
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    test_res = extract_split_logits(
        model=model,
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    cache_payload = {
        "val_logits": val_res["logits"],
        "val_targets": val_res["targets"],
        "val_paths": val_res["paths"],
        "val_accuracy": val_res["accuracy"],
        "val_macro_f1": val_res["macro_f1"],
        "test_logits": test_res["logits"],
        "test_targets": test_res["targets"],
        "test_paths": test_res["paths"],
        "test_accuracy": test_res["accuracy"],
        "test_macro_f1": test_res["macro_f1"],
        "classes": val_dataset.classes,
        "class_to_idx": class_to_idx,
        "checkpoint_epoch": ckpt_epoch,
        "checkpoint_path": str(checkpoint_path),
    }

    output_cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_cache_path.with_suffix(".tmp.pt")
    torch.save(cache_payload, tmp_path)
    os.replace(tmp_path, output_cache_path)
    logger.info(f"Saved logits cache atomically to: {output_cache_path}")

    return cache_payload


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Validation and Test Logits from Best DINOv2-LoRA Checkpoint")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=r"c:\CropGuard v3\augmented_plantdoc_clean",
        help="Path to sanitized dataset directory",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=r"c:\CropGuard v3\checkpoints\dinov2_lora\best_checkpoint.pt",
        help="Path to best model checkpoint",
    )
    parser.add_argument(
        "--output-cache",
        type=str,
        default=r"c:\CropGuard v3\checkpoints\dinov2_lora\logits_cache.pt",
        help="Output path for cached logits tensor dictionary",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=47, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_logit_extraction(
        data_dir=Path(args.data_dir),
        checkpoint_path=Path(args.checkpoint_path),
        output_cache_path=Path(args.output_cache),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
