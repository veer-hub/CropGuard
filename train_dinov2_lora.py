"""
DINOv2-Large + LoRA Training Pipeline with Multi-Head Attentive Pathology Pooling (MAF)
CropGuard v3 - Autonomous High-Performance Crop Disease Classification

Hardware Configuration:
- GPU: NVIDIA GeForce RTX 3070 (8 GB VRAM), CUDA active
- Precision: Mixed precision fp16 (torch.amp.autocast)
- Effective Batch Size: 32 (micro-batch size 16 x gradient accumulation steps 2)
- Architecture: DINOv2-Large (304M frozen) + LoRA (r=16, alpha=32) + MAF Pooling + Linear Head (27 classes)
"""

import os
import sys
import math
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score, accuracy_score, classification_report

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add parent directory for utils import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.checkpoint_manager import CheckpointManager

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("CropGuardTrainer")


# ==============================================================================
# 1. Dataset & Data Augmentation
# ==============================================================================

def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Dynamic Albumentations augmentations for in-the-wild agricultural robustness."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=0,
            p=0.5,
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Validation and testing deterministic transforms."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


class PlantDocCleanDataset(Dataset):
    """Zero-leakage PlantDoc dataset loader supporting Albumentations transforms."""

    def __init__(
        self,
        split_dir: Path,
        transform: Optional[A.Compose] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.split_dir = Path(split_dir)
        self.transform = transform
        
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory does not exist: {self.split_dir}")

        # Scan classes and build mapping
        subdirs = sorted([d.name for d in self.split_dir.iterdir() if d.is_dir()])
        if class_to_idx is None:
            self.classes = subdirs
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        else:
            self.classes = sorted(list(class_to_idx.keys()))
            self.class_to_idx = class_to_idx

        # Index image paths and labels
        self.samples: List[Tuple[Path, int]] = []
        for cname in self.classes:
            cdir = self.split_dir / cname
            if not cdir.exists():
                continue
            cidx = self.class_to_idx[cname]
            for f in cdir.iterdir():
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.samples.append((f, cidx))

        logger.info(
            f"Loaded {len(self.samples)} images from {self.split_dir.name} "
            f"across {len(self.classes)} classes."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img_np = np.array(img)

        if self.transform is not None:
            augmented = self.transform(image=img_np)
            tensor = augmented["image"]
        else:
            # Fallback simple tensor conversion
            tensor = torch.from_numpy(img_np.transpose((2, 0, 1))).float() / 255.0

        return tensor, label


# ==============================================================================
# 2. Multi-Head Attentive Pathology Pooling (MAF)
# ==============================================================================

class MultiHeadAttentivePooling(nn.Module):
    """Multi-Head Attentive Pathology Pooling (MAF).
    
    Computes dynamic attention weights over localized patch tokens:
        alpha_i = softmax(q^T W_k h_i / sqrt(d_k))
        h_lesion = sum_i alpha_i (W_v h_i)
    
    This mathematically suppresses extraneous background clutter (soil, fruit, bench)
    and isolates disease morphology without losing raw token resolution.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        # Learned pathology query vector across heads
        self.query = nn.Parameter(torch.randn(1, num_heads, 1, self.head_dim) * 0.02)

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            patch_tokens: Tensor of shape (B, N, D), where N is number of patch tokens (256)
            return_attention: If True, returns attention distribution across patches
        Returns:
            pooled: Tensor of shape (B, D)
            attn_weights: Tensor of shape (B, num_heads, N) if return_attention else None
        """
        B, N, D = patch_tokens.shape

        # Linear projections
        # (B, N, D) -> (B, N, H, d_k) -> (B, H, N, d_k)
        K = self.k_proj(patch_tokens).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(patch_tokens).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Pathology query expanded across batch: (B, H, 1, d_k)
        Q = self.query.expand(B, -1, -1, -1)

        # Scaled dot-product attention scores: (B, H, 1, N)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_dropped = self.dropout(attn_weights)

        # Aggregated lesion representation: (B, H, 1, d_k)
        pooled_heads = torch.matmul(attn_dropped, V)

        # Concatenate heads: (B, 1, H * d_k) -> (B, D)
        pooled = pooled_heads.transpose(1, 2).contiguous().view(B, D)
        pooled = self.norm(self.out_proj(pooled))

        if return_attention:
            return pooled, attn_weights.squeeze(2)
        return pooled, None


# ==============================================================================
# 3. Complete DINOv2 + LoRA + MAF Architecture
# ==============================================================================

class DinoV2LoRAPathologyModel(nn.Module):
    """DINOv2-Large backbone with Low-Rank Adaptation and MAF pooling."""

    def __init__(
        self,
        backbone_id: str = "facebook/dinov2-large",
        num_classes: int = 27,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        pooling_heads: int = 8,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        logger.info(f"Loading pre-trained foundation backbone: {backbone_id}")
        base_backbone = AutoModel.from_pretrained(backbone_id)
        embed_dim = base_backbone.config.hidden_size  # 1024 for ViT-Large

        # Freeze the entire 304M parameter backbone
        for param in base_backbone.parameters():
            param.requires_grad = False

        # Attach LoRA to query, key, value projection matrices
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj"],
            lora_dropout=lora_dropout,
            bias="none",
        )
        self.backbone = get_peft_model(base_backbone, lora_config)

        # Multi-Head Attentive Pathology Pooling over patch tokens
        self.maf_pooling = MultiHeadAttentivePooling(
            embed_dim=embed_dim,
            num_heads=pooling_heads,
            dropout=head_dropout,
        )

        # Classification Head (27 classes)
        self.classifier = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self._log_parameter_summary()

    def _log_parameter_summary(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"Model Parameters: Trainable = {trainable_params:,} "
            f"({(trainable_params / total_params) * 100:.2f}%) | "
            f"Total = {total_params:,}"
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        Args:
            pixel_values: Input image tensor (B, 3, 224, 224)
            return_attention: If True, returns MAF attention maps
        Returns:
            logits: Classification logits (B, num_classes)
            attention: Optional attention tensor (B, num_heads, N)
        """
        outputs = self.backbone(pixel_values=pixel_values)
        # last_hidden_state: (B, 1 + N, D)
        # Token 0 is [CLS]; tokens 1: are patch tokens (16x16 = 256 patches)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]

        # Multi-head attentive pooling over lesion patches
        pooled, attn = self.maf_pooling(patch_tokens, return_attention=return_attention)

        # Class logits
        logits = self.classifier(pooled)
        return logits, attn


# ==============================================================================
# 4. LR Scheduler: 3 Warmup + 20 Cosine Decay Epochs
# ==============================================================================

def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    lr_min: float = 1e-6,
    lr_max: float = 2e-4,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Builds a learning rate scheduler with 3 linear warmup epochs and 20 cosine decay epochs."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup from lr_min to lr_max
            alpha = step / max(1, warmup_steps)
            current_lr = lr_min + (lr_max - lr_min) * alpha
        else:
            # Cosine annealing decay from lr_max down to lr_min
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            current_lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))
        # LambdaLR scales base_lr (which is set to lr_max in optimizer)
        return current_lr / lr_max

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ==============================================================================
# 5. Training & Validation Engines
# ==============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_accum_steps: int = 2,
    epoch: int = 1,
    max_steps: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Runs one training epoch with mixed precision and gradient accumulation."""
    model.train()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    optimizer.zero_grad(set_to_none=True)
    num_batches = len(loader) if max_steps is None else min(len(loader), max_steps)
    start_time = time.time()

    for step, (images, targets) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            logits, _ = model(images)
            loss = criterion(logits, targets)
            loss_scaled = loss / grad_accum_steps

        scaler.scale(loss_scaled).backward()
        total_loss += loss.item() * targets.size(0)

        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.detach().cpu().numpy())

        # Optimizer step upon accumulating gradients or on final batch
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == num_batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        if (step + 1) % 50 == 0 or (step + 1) == num_batches:
            current_lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start_time
            logger.info(
                f"Epoch [{epoch:02d}] Step [{step + 1:04d}/{num_batches:04d}] "
                f"Loss: {loss.item():.4f} | LR: {current_lr:.2e} | Elapsed: {elapsed:.1f}s"
            )

    avg_loss = total_loss / max(1, len(all_targets))
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_steps: Optional[int] = None,
) -> Tuple[float, float, float, Dict[str, Any]]:
    """Evaluates the model on validation or test split and computes Macro-F1."""
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    for step, (images, targets) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            logits, _ = model(images)
            loss = criterion(logits, targets)

        total_loss += loss.item() * targets.size(0)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / max(1, len(all_targets))
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    report = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)

    return avg_loss, acc, macro_f1, report


# ==============================================================================
# 6. Main Orchestrator
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="CropGuard v3 - DINOv2-Large LoRA Training Pipeline")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=r"c:\CropGuard v3\augmented_plantdoc_clean",
        help="Path to sanitized zero-leakage dataset directory",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=r"c:\CropGuard v3\checkpoints",
        help="Root directory for saving checkpoints",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="dinov2_lora",
        help="Subdirectory for this specific experiment",
    )
    parser.add_argument("--epochs", type=int, default=23, help="Total training epochs (default: 23)")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Warmup epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=16, help="Micro-batch size (default: 16)")
    parser.add_argument("--grad-accum-steps", type=int, default=2, help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--lr-max", type=float, default=2e-4, help="Peak learning rate (default: 2e-4)")
    parser.add_argument("--lr-min", type=float, default=1e-6, help="Minimum learning rate (default: 1e-6)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank r")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha scaling parameter")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--pooling-heads", type=int, default=8, help="Number of MAF attention heads")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes")
    parser.add_argument("--seed", type=int, default=47, help="Random reproducibility seed")
    parser.add_argument("--resume", action="store_true", help="Resume from latest crash-safe checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 quick step for pipeline validation")
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    # Datasets
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    train_dataset = PlantDocCleanDataset(train_dir, transform=train_transform)
    val_dataset = PlantDocCleanDataset(val_dir, transform=val_transform, class_to_idx=train_dataset.class_to_idx)
    test_dataset = PlantDocCleanDataset(test_dir, transform=val_transform, class_to_idx=train_dataset.class_to_idx)

    num_classes = len(train_dataset.classes)
    logger.info(f"Target classification classes: {num_classes}")

    # DataLoaders
    # Micro-batch size 16, effective batch size 32 via grad_accum_steps 2
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Model instantiation
    model = DinoV2LoRAPathologyModel(
        num_classes=num_classes,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        pooling_heads=args.pooling_heads,
    ).to(device)

    # Optimizer: Only trainable parameters (LoRA + MAF + Head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr_max,
        weight_decay=args.weight_decay,
    )

    # Scheduler: 3 linear warmup epochs, 20 cosine annealing decay epochs
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    scheduler = build_lr_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        lr_min=args.lr_min,
        lr_max=args.lr_max,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler(device="cuda", enabled=(device.type == "cuda"))

    # Crash-safe Checkpoint Manager
    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=args.experiment_name,
    )

    start_epoch = 1
    best_val_f1 = 0.0

    # Resume capability
    if args.resume and checkpoint_mgr.latest_path.exists():
        logger.info(f"Attempting to resume from: {checkpoint_mgr.latest_path}")
        checkpoint = torch.load(checkpoint_mgr.latest_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "extra_state" in checkpoint and "scaler_state_dict" in checkpoint["extra_state"]:
            scaler.load_state_dict(checkpoint["extra_state"]["scaler_state_dict"])
        
        start_epoch = checkpoint["epoch"] + 1
        best_val_f1 = checkpoint.get("extra_state", {}).get("best_val_f1", checkpoint.get("val_metric", 0.0))
        logger.info(f"Resumed successfully! Next epoch: {start_epoch}, Prior Best Val Macro-F1: {best_val_f1:.4f}")

    if args.dry_run:
        logger.info("=== RUNNING DRY RUN SANITY CHECK (1 train step, 1 val step) ===")
        train_loss, train_acc, train_f1 = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            grad_accum_steps=1,
            epoch=1,
            max_steps=2,
        )
        val_loss, val_acc, val_f1, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            max_steps=2,
        )
        logger.info(
            f"[Dry Run Sanity] Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}"
        )
        # Test checkpoint saving
        checkpoint_mgr.save_checkpoint(
            epoch=1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            val_metric=val_f1,
            is_best=True,
            extra_state={"best_val_f1": val_f1, "classes": train_dataset.classes},
        )
        checkpoint_mgr.log_metrics(1, {"val_loss": val_loss, "val_macro_f1": val_f1, "is_best": True})
        logger.info("=== DRY RUN COMPLETED SUCCESSFULLY ===")
        return

    # ==========================================================================
    # 7. Main Training Loop: 23 Epochs with Peak Saving Every Epoch
    # ==========================================================================
    logger.info("================================================================================")
    logger.info(f"Starting Training: Total Epochs = {args.epochs} | Warmup = {args.warmup_epochs}")
    logger.info(f"Micro-batch Size = {args.batch_size} | Gradient Accumulation = {args.grad_accum_steps} | Effective Batch Size = {args.batch_size * args.grad_accum_steps}")
    logger.info(f"Peak LR = {args.lr_max} | Min LR = {args.lr_min} | Device = {device}")
    logger.info("================================================================================")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        logger.info(f"\n--- Epoch [{epoch:02d}/{args.epochs:02d}] ---")

        # 1. Train 1 full epoch
        train_loss, train_acc, train_f1 = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            grad_accum_steps=args.grad_accum_steps,
            epoch=epoch,
        )

        # 2. Evaluate on validation set at EVERY epoch
        val_loss, val_acc, val_f1, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        epoch_duration = time.time() - epoch_start

        # 3. Check for new peak
        is_best = bool(val_f1 > best_val_f1)
        if is_best:
            best_val_f1 = float(val_f1)
            logger.info(
                f"[PEAK ACHIEVED!] Epoch {epoch:02d}: Val Macro-F1 reached NEW BEST: "
                f"{val_f1:.4f} (Accuracy: {val_acc * 100:.2f}%)"
            )
        else:
            logger.info(
                f"Epoch {epoch:02d}: Val Macro-F1 = {val_f1:.4f} "
                f"(Best so far: {best_val_f1:.4f})"
            )

        # 4. Save checkpoint:
        # - Saves epoch checkpoint atomically
        # - Atomically updates latest_checkpoint.pt EVERY epoch
        # - Atomically updates best_checkpoint.pt if peak was achieved!
        extra_state = {
            "best_val_f1": float(best_val_f1),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "train_macro_f1": float(train_f1),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_macro_f1": float(val_f1),
            "scaler_state_dict": scaler.state_dict(),
            "classes": train_dataset.classes,
            "epoch_duration_seconds": float(epoch_duration),
        }
        saved_path = checkpoint_mgr.save_checkpoint(
            epoch=int(epoch),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            val_metric=float(val_f1),
            is_best=is_best,
            extra_state=extra_state,
        )

        # 5. Flush metric record to JSONL history file
        checkpoint_mgr.log_metrics(
            epoch=int(epoch),
            metrics={
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "train_macro_f1": float(train_f1),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "val_macro_f1": float(val_f1),
                "is_best": is_best,
                "best_val_f1": float(best_val_f1),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "duration_seconds": float(epoch_duration),
            },
        )

        logger.info(
            f"Epoch [{epoch:02d}] Summary: Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f} | Duration: {epoch_duration:.1f}s"
        )

    # Final evaluation on test set using best checkpoint
    logger.info("\n================================================================================")
    logger.info("Training complete. Evaluating BEST checkpoint on held-out test split...")
    logger.info("================================================================================")

    if checkpoint_mgr.best_path.exists():
        logger.info(f"Loading best checkpoint from: {checkpoint_mgr.best_path}")
        best_ckpt = torch.load(checkpoint_mgr.best_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        test_loss, test_acc, test_f1, test_report = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )
        logger.info(
            f"[TEST BENCHMARK RESULTS] Test Loss: {test_loss:.4f} | "
            f"Test Accuracy: {test_acc * 100:.2f}% | Test Macro-F1: {test_f1:.4f}"
        )
        checkpoint_mgr.log_metrics(
            epoch=args.epochs,
            metrics={
                "final_test_loss": test_loss,
                "final_test_acc": test_acc,
                "final_test_macro_f1": test_f1,
            },
        )


if __name__ == "__main__":
    main()
