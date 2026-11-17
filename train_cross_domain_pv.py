"""
Phase 5 - Step 5.2: Cross-Domain Lab-to-Field Transfer Training Engine
CropGuard v3 - PlantVillage (PV_200 & PV_100) -> PlantDoc Zero-Shot Transfer

Implements a 3-stage cross-domain ablation pipeline for both PV_200 and PV_100:
  - Stage A (Source-Only ERM Baseline): Trained strictly on raw laboratory PlantVillage images.
  - Stage B (Source-Only + Segmented Background Desensitization + MAF): Composites segmented
    foreground leaves onto procedurally generated field-like backgrounds + strong photometric/
    geometric jitter to break the laboratory studio background shortcut (100% source-only).
  - Stage C (Full CropGuard v3 Domain-Adaptive Transfer): Combines Stage B source supervision
    with Deep CORAL feature covariance alignment, SHOT Information Maximization (conditional
    entropy minimization + batch marginal uniformity), and confidence-gated weak-to-strong
    self-training on UNLABELED PlantDoc training images (val & test remain 100% unseen).
"""

import os
import gc
import sys
import json
import math
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_dinov2_lora import (
    DinoV2LoRAPathologyModel,
    _TorchvisionAlbumentationsWrapper,
    build_lr_scheduler,
    get_val_transforms,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("CrossDomainTrainer")

BASE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = BASE_DIR / "docs" / "pv_subsets_manifest.json"
PD_TRAIN_DIR = BASE_DIR / "augmented_plantdoc_clean" / "train"
CKPT_DIR = BASE_DIR / "checkpoints" / "cross_domain_pv"


def atomic_torch_save(state: Dict[str, Any], target_path: Path) -> None:
    """Atomically saves a PyTorch checkpoint using .tmp + fsync + os.replace."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(".pt.tmp")
    with open(tmp_path, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, target_path)


def generate_synthetic_field_bg(size: int, rng: np.random.Generator) -> np.ndarray:
    """Generates a multi-frequency natural agricultural background canvas (soil, canopy, sky, shade)."""
    # Choose agricultural background palette: earth soil, blurred foliage, overcast sky, bark/mulch
    palettes = [
        ([95, 76, 56], [58, 44, 30]),     # Brown soil / mulch
        ([64, 102, 48], [38, 66, 26]),    # Blurred green canopy bokeh
        ([112, 128, 86], [78, 92, 55]),   # Dry grass / mixed field
        ([145, 155, 160], [95, 105, 112]),# Overcast outdoor sky / shade
        ([82, 92, 68], [130, 112, 84]),   # Soil + foliage mix
    ]
    c1, c2 = palettes[int(rng.integers(0, len(palettes)))]
    c1 = np.array(c1, dtype=np.float32) + rng.uniform(-18, 18, size=3)
    c2 = np.array(c2, dtype=np.float32) + rng.uniform(-18, 18, size=3)

    # Smooth spatial gradient + low-frequency cloud blobs
    gx = np.linspace(0.0, 1.0, size, dtype=np.float32)
    gy = np.linspace(0.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(gx, gy)
    angle = rng.uniform(0, 2 * math.pi)
    grad = 0.5 + 0.5 * np.sin((xx * math.cos(angle) + yy * math.sin(angle)) * math.pi)

    low_res = rng.uniform(0.0, 1.0, size=(8, 8)).astype(np.float32)
    blobs = cv2.resize(low_res, (size, size), interpolation=cv2.INTER_CUBIC)
    mix = np.clip(0.55 * grad + 0.45 * blobs, 0.0, 1.0)[..., None]

    bg = c1 * mix + c2 * (1.0 - mix)

    # Add fine organic soil/texture speckle
    speckle = rng.normal(0.0, rng.uniform(6.0, 18.0), size=(size // 2, size // 2, 3)).astype(np.float32)
    speckle_up = cv2.resize(speckle, (size, size), interpolation=cv2.INTER_LINEAR)
    bg = np.clip(bg + speckle_up, 0, 255).astype(np.uint8)
    return bg


def composite_segmented_leaf(
    color_path: str,
    segmented_path: Optional[str],
    rng: np.random.Generator,
    canvas_size: int = 256,
) -> np.ndarray:
    """Composites a segmented leaf cutout onto a randomized non-uniform background."""
    if segmented_path is None or not Path(segmented_path).exists() or rng.random() < 0.20:
        with Image.open(color_path) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)

    with Image.open(segmented_path) as seg_img:
        seg_np = np.asarray(seg_img.convert("RGB"), dtype=np.uint8)

    h, w, _ = seg_np.shape
    if h != canvas_size or w != canvas_size:
        seg_np = cv2.resize(seg_np, (canvas_size, canvas_size), interpolation=cv2.INTER_LINEAR)

    # Build feathered foreground alpha mask excluding pure black background
    fg_raw = (seg_np.max(axis=-1) > 14).astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)
    fg_eroded = cv2.erode(fg_raw, kernel, iterations=1)
    alpha = cv2.GaussianBlur(fg_eroded, (5, 5), sigmaX=1.2)

    # Random leaf scale and shift on canvas to mimic variable field capture distances
    scale = float(rng.uniform(0.60, 1.15))
    tx = float(rng.uniform(-0.14, 0.14) * canvas_size)
    ty = float(rng.uniform(-0.14, 0.14) * canvas_size)
    rot_deg = float(rng.uniform(-35.0, 35.0))
    M = cv2.getRotationMatrix2D((canvas_size / 2.0, canvas_size / 2.0), rot_deg, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    seg_warped = cv2.warpAffine(seg_np, M, (canvas_size, canvas_size), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    alpha_warped = cv2.warpAffine(alpha, M, (canvas_size, canvas_size), flags=cv2.INTER_LINEAR, borderValue=0.0)[..., None]

    bg = generate_synthetic_field_bg(canvas_size, rng)
    composited = seg_warped.astype(np.float32) * alpha_warped + bg.astype(np.float32) * (1.0 - alpha_warped)
    return np.clip(composited, 0, 255).astype(np.uint8)


def get_source_transforms(stage: str, image_size: int = 224) -> Any:
    """Returns torchvision transforms wrapped identically for Stage A vs Stage B/C."""
    if stage == "a":
        return _TorchvisionAlbumentationsWrapper(T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))
    return _TorchvisionAlbumentationsWrapper(T.Compose([
        T.RandomResizedCrop(size=(image_size, image_size), scale=(0.55, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(brightness=0.30, contrast=0.30, saturation=0.30, hue=0.08)], p=0.8),
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.25),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))


def get_target_weak_strong_transforms(image_size: int = 224) -> Tuple[Any, Any]:
    """Returns weak and strong augmentation pipelines for unlabeled target self-training."""
    weak = _TorchvisionAlbumentationsWrapper(T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    strong = _TorchvisionAlbumentationsWrapper(T.Compose([
        T.RandomResizedCrop(size=(image_size, image_size), scale=(0.55, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.08)], p=0.8),
        T.RandomApply([T.RandomAffine(degrees=20, scale=(0.9, 1.1))], p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    return weak, strong


class PVManifestDataset(Dataset):
    """Dataset backed by pv_subsets_manifest.json supporting raw or segmented-randomized modes."""

    def __init__(
        self,
        records: List[Dict[str, Any]],
        transform: Any,
        randomize_bg: bool = False,
        seed: int = 47,
    ):
        self.records = records
        self.transform = transform
        self.randomize_bg = randomize_bg
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rec = self.records[idx]
        if self.randomize_bg:
            rng = np.random.default_rng()
            img_np = composite_segmented_leaf(
                color_path=rec["color_path"],
                segmented_path=rec.get("segmented_path"),
                rng=rng,
            )
        else:
            with Image.open(rec["color_path"]) as img:
                img_np = np.asarray(img.convert("RGB"), dtype=np.uint8)

        tensor = self.transform(image=img_np)["image"]
        return tensor, int(rec["class_idx"])


class UnlabeledTargetFieldDataset(Dataset):
    """Loads ONLY unlabeled images from augmented_plantdoc_clean/train (zero labels, zero val/test)."""

    def __init__(self, train_dir: Path, weak_transform: Any, strong_transform: Any):
        if train_dir.name != "train":
            raise ValueError(f"Unlabeled target dataset must point strictly to 'train', got {train_dir}")
        if not train_dir.exists():
            raise FileNotFoundError(f"Target train directory not found: {train_dir}")

        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.image_paths: List[Path] = []
        for cdir in sorted(train_dir.iterdir()):
            if cdir.is_dir():
                for f in sorted(cdir.iterdir()):
                    if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        self.image_paths.append(f)

        logger.info(
            f"Loaded {len(self.image_paths)} UNLABELED target field images strictly from {train_dir} "
            f"(val/test 100% excluded)."
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.image_paths[idx]) as img:
            img_np = np.asarray(img.convert("RGB"), dtype=np.uint8)
        weak_t = self.weak_transform(image=img_np)["image"]
        strong_t = self.strong_transform(image=img_np)["image"]
        return weak_t, strong_t


def deep_coral_loss(source_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
    """Computes Deep CORAL covariance + mean alignment loss between source and target embeddings."""
    d = source_feat.size(1)
    ns, nt = source_feat.size(0), target_feat.size(0)
    if ns < 2 or nt < 2:
        return torch.tensor(0.0, device=source_feat.device)

    s_f = source_feat.float()
    t_f = target_feat.float()

    # Mean alignment
    mean_s = s_f.mean(dim=0, keepdim=True)
    mean_t = t_f.mean(dim=0, keepdim=True)
    mean_loss = F.mse_loss(mean_s, mean_t)

    # Covariance matrices
    s_centered = s_f - mean_s
    t_centered = t_f - mean_t
    cov_s = (s_centered.t() @ s_centered) / max(1, ns - 1)
    cov_t = (t_centered.t() @ t_centered) / max(1, nt - 1)

    coral = torch.sum((cov_s - cov_t) ** 2) / (4.0 * d * d)
    return coral + 0.5 * mean_loss


def shot_information_maximization_loss(logits_t: torch.Tensor) -> torch.Tensor:
    """SHOT Information Maximization: conditional entropy minimization - 2.0 * marginal entropy."""
    probs = F.softmax(logits_t.float(), dim=-1)
    eps = 1e-6
    cond_entropy = -torch.mean(torch.sum(probs * torch.log(probs + eps), dim=-1))
    mean_prob = torch.mean(probs, dim=0)
    marginal_entropy = -torch.sum(mean_prob * torch.log(mean_prob + eps))
    return cond_entropy - 2.0 * marginal_entropy


def forward_with_features(
    model: DinoV2LoRAPathologyModel,
    pixel_values: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass returning both classification logits and pooled MAF embeddings."""
    outputs = model.backbone(pixel_values=pixel_values)
    patch_tokens = outputs.last_hidden_state[:, 1:, :]
    pooled, _ = model.maf_pooling(patch_tokens, return_attention=False)
    logits = model.classifier(pooled)
    return logits, pooled


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluates accuracy, macro-F1, and loss on a labeled validation/test loader."""
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []
    total_loss = 0.0
    total_n = 0
    criterion = nn.CrossEntropyLoss()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits, _ = model(images, return_attention=False)
            loss = criterion(logits, targets)
        total_loss += float(loss.item()) * images.size(0)
        total_n += images.size(0)
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    acc = float(accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    return total_loss / max(1, total_n), acc, macro_f1


def train_source_stage(
    subset_name: str,
    stage: str,
    train_records: List[Dict[str, Any]],
    val_loader: DataLoader,
    classes: List[str],
    epochs: int,
    lr_max: float,
    batch_size: int,
    grad_accum_steps: int,
    num_workers: int,
    device: torch.device,
) -> Path:
    """Trains Stage A (raw lab) or Stage B (segmented background randomization + MAF)."""
    assert stage in ("a", "b")
    randomize_bg = (stage == "b")
    save_path = CKPT_DIR / f"{subset_name}_stage_{stage}_best.pt"
    logger.info("=" * 75)
    logger.info(f"Starting {subset_name.upper()} Stage {stage.upper()} | Epochs={epochs} | RandomizeBG={randomize_bg}")
    logger.info("=" * 75)

    train_ds = PVManifestDataset(
        records=train_records,
        transform=get_source_transforms(stage),
        randomize_bg=randomize_bg,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )

    model = DinoV2LoRAPathologyModel(num_classes=len(classes)).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr_max, weight_decay=0.02)

    steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * 1
    scheduler = build_lr_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps, lr_min=1e-6, lr_max=lr_max)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler(device="cuda", enabled=(device.type == "cuda"))

    best_val_f1 = -1.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        n_batches = len(train_loader)

        for step_idx, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits, _ = model(images, return_attention=False)
                loss = criterion(logits, targets) / grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += float(loss.item()) * grad_accum_steps

            if step_idx % grad_accum_steps == 0 or step_idx == n_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        val_loss, val_acc, val_f1 = evaluate_loader(model, val_loader, device)
        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            atomic_torch_save(
                {
                    "subset": subset_name,
                    "stage": stage,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "pv_val_acc": val_acc,
                    "pv_val_macro_f1": val_f1,
                    "classes": classes,
                },
                save_path,
            )

        logger.info(
            f"[{subset_name.upper()} Stage {stage.upper()}] Epoch {epoch:02d}/{epochs:02d} | "
            f"TrainLoss={running_loss / max(1, n_batches):.4f} | PV_Val Acc={val_acc * 100:.2f}% | "
            f"PV_Val F1={val_f1:.4f} {'[BEST]' if is_best else ''} | Time={time.time() - t0:.1f}s"
        )

    del model, optimizer, scaler, scheduler, train_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return save_path


@torch.no_grad()
def update_ema_dict(ema_params: Dict[str, torch.Tensor], model: nn.Module, decay: float = 0.992) -> None:
    """Updates lightweight EMA dictionary containing ONLY trainable parameters (~22 MB)."""
    for name, param in model.named_parameters():
        if param.requires_grad and name in ema_params:
            ema_params[name].mul_(decay).add_(param.data, alpha=1.0 - decay)


@torch.no_grad()
def swap_trainable_params(model: nn.Module, target_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Swaps trainable model parameters with target_params in-place and returns the previous values."""
    backup: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.requires_grad and name in target_params:
            backup[name] = param.data.detach().clone()
            param.data.copy_(target_params[name])
    return backup


def train_uda_stage_c(
    subset_name: str,
    stage_b_ckpt_path: Path,
    train_records: List[Dict[str, Any]],
    val_loader: DataLoader,
    classes: List[str],
    epochs: int,
    lr_max: float,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Path:
    """Trains Stage C: Deep CORAL + SHOT IM + Confidence-Gated Weak-to-Strong UDA on unlabeled train/.

    Memory-optimized for 8 GB VRAM:
      - Stores EMA only for the ~5.5M trainable LoRA/MAF weights (~22 MB vs 1.24 GB full model copy).
      - Backpropagates source and target graphs sequentially so only 1 ViT-Large graph resides in VRAM.
    """
    save_path = CKPT_DIR / f"{subset_name}_stage_c_best.pt"
    alias_path = CKPT_DIR / f"{subset_name.replace('_', '')}_best.pt"

    logger.info("=" * 75)
    logger.info(f"Starting {subset_name.upper()} Stage C (Deep CORAL + SHOT IM + FixMatch UDA, Low-VRAM) | Epochs={epochs}")
    logger.info("=" * 75)

    source_ds = PVManifestDataset(
        records=train_records,
        transform=get_source_transforms("b"),
        randomize_bg=True,
    )
    weak_tf, strong_tf = get_target_weak_strong_transforms()
    target_ds = UnlabeledTargetFieldDataset(PD_TRAIN_DIR, weak_tf, strong_tf)

    uda_workers = min(2, num_workers)
    source_loader = DataLoader(
        source_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=uda_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
        drop_last=True,
    )
    target_loader = DataLoader(
        target_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=uda_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
        drop_last=True,
    )

    model = DinoV2LoRAPathologyModel(num_classes=len(classes)).to(device)
    ckpt = torch.load(stage_b_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Warm-started Stage C from Stage B checkpoint: {stage_b_ckpt_path} (PV_Val F1={ckpt['pv_val_macro_f1']:.4f})")

    ema_params: Dict[str, torch.Tensor] = {
        name: param.data.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr_max, weight_decay=0.02)

    steps_per_epoch = max(len(source_loader), len(target_loader))
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * 1
    scheduler = build_lr_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps, lr_min=1e-6, lr_max=lr_max)
    sup_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler(device="cuda", enabled=(device.type == "cuda"))

    num_classes = len(classes)
    ema_marginal = torch.full((num_classes,), 1.0 / num_classes, device=device)
    src_protos = torch.zeros((num_classes, 1024), device=device)
    proto_counts = torch.zeros(num_classes, device=device)

    best_score = -1.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        src_iter = iter(source_loader)
        tgt_iter = iter(target_loader)

        sup_sum = 0.0
        coral_sum = 0.0
        im_sum = 0.0
        pseudo_sum = 0.0
        accepted_frac_sum = 0.0

        ramp = min(1.0, epoch / max(1.0, epochs * 0.5))

        for step in range(1, steps_per_epoch + 1):
            try:
                src_x, src_y = next(src_iter)
            except StopIteration:
                src_iter = iter(source_loader)
                src_x, src_y = next(src_iter)

            try:
                tgt_weak, tgt_strong = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(target_loader)
                tgt_weak, tgt_strong = next(tgt_iter)

            src_x = src_x.to(device, non_blocking=True)
            src_y = src_y.to(device, non_blocking=True)
            tgt_weak = tgt_weak.to(device, non_blocking=True)
            tgt_strong = tgt_strong.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Pass 1 (no-grad): Distribution-Aligned EMA teacher pseudo-labels + detached target feature reference
            model.eval()
            with torch.no_grad():
                train_backup = swap_trainable_params(model, ema_params)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    ema_logits, _ = model(tgt_weak, return_attention=False)
                swap_trainable_params(model, train_backup)

                ema_probs = F.softmax(ema_logits.float(), dim=-1)
                ema_marginal = 0.99 * ema_marginal + 0.01 * ema_probs.mean(dim=0)
                da_probs = ema_probs / (ema_marginal.unsqueeze(0) + 1e-6)
                da_probs = da_probs / da_probs.sum(dim=-1, keepdim=True)
                max_probs, pseudo_labels = torch.max(da_probs, dim=-1)
                conf_mask = max_probs >= 0.65

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    _, feat_t_ref = forward_with_features(model, tgt_strong)
                feat_t_ref = feat_t_ref.detach().float()

            model.train()

            # Pass 2 (source backward -> frees source autograd graph immediately)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits_s, feat_s = forward_with_features(model, src_x)
                loss_sup = sup_criterion(logits_s, src_y)
                loss_coral_s = deep_coral_loss(feat_s, feat_t_ref)
                loss_source_total = 1.25 * loss_sup + (0.45 * ramp) * loss_coral_s
            feat_s_ref = feat_s.detach().float()
            with torch.no_grad():
                for c_id in torch.unique(src_y):
                    c_mask = (src_y == c_id)
                    c_mean = F.normalize(feat_s_ref[c_mask].mean(dim=0), dim=-1)
                    if proto_counts[c_id] == 0:
                        src_protos[c_id] = c_mean
                    else:
                        src_protos[c_id] = F.normalize(0.9 * src_protos[c_id] + 0.1 * c_mean, dim=-1)
                    proto_counts[c_id] += 1
            scaler.scale(loss_source_total).backward()

            # Pass 3 (target backward -> frees target autograd graph immediately)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits_t, feat_t = forward_with_features(model, tgt_strong)
                loss_coral_t = deep_coral_loss(feat_s_ref, feat_t)
                loss_im = shot_information_maximization_loss(logits_t)
                if conf_mask.any():
                    loss_pseudo = (
                        F.cross_entropy(logits_t, pseudo_labels, reduction="none", label_smoothing=0.10)
                        * conf_mask.float()
                    ).mean()
                    # Prototype cosine alignment for confident target samples
                    valid_p = proto_counts[pseudo_labels[conf_mask]] > 0
                    if valid_p.any():
                        t_norm = F.normalize(feat_t[conf_mask][valid_p].float(), dim=-1)
                        p_target = src_protos[pseudo_labels[conf_mask][valid_p]]
                        loss_proto = (1.0 - torch.sum(t_norm * p_target, dim=-1)).mean()
                    else:
                        loss_proto = torch.tensor(0.0, device=device)
                else:
                    loss_pseudo = torch.tensor(0.0, device=device)
                    loss_proto = torch.tensor(0.0, device=device)
                loss_target_total = (
                    (0.45 * ramp) * loss_coral_t
                    + (0.35 * ramp) * loss_im
                    + (0.35 * ramp) * loss_pseudo
                    + (0.30 * ramp) * loss_proto
                )
            scaler.scale(loss_target_total).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            update_ema_dict(ema_params, model, decay=0.992)

            sup_sum += float(loss_sup.item())
            coral_sum += float(loss_coral_t.item())
            im_sum += float(loss_im.item())
            pseudo_sum += float(loss_pseudo.item())
            accepted_frac_sum += float(conf_mask.float().mean().item())

            if step % 100 == 0 or step == steps_per_epoch:
                logger.info(
                    f"  [{subset_name.upper()} Stage C] Ep {epoch:02d} Step [{step:03d}/{steps_per_epoch:03d}] | "
                    f"Sup={sup_sum/step:.4f} | CORAL={coral_sum/step:.4f} | Elapsed={time.time()-t0:.1f}s"
                )

        # Evaluate EMA weights strictly on PV_Val (zero target test leakage!)
        train_backup = swap_trainable_params(model, ema_params)
        val_loss, val_acc, val_f1 = evaluate_loader(model, val_loader, device)
        is_best = val_f1 >= best_score - 0.005
        if val_f1 > best_score:
            best_score = val_f1

        if is_best or epoch == epochs:
            state = {
                "subset": subset_name,
                "stage": "c",
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "pv_val_acc": val_acc,
                "pv_val_macro_f1": val_f1,
                "classes": classes,
            }
            atomic_torch_save(state, save_path)
            atomic_torch_save(state, alias_path)

        swap_trainable_params(model, train_backup)

        logger.info(
            f"[{subset_name.upper()} Stage C] Epoch {epoch:02d}/{epochs:02d} | "
            f"Sup={sup_sum/steps_per_epoch:.4f} | CORAL={coral_sum/steps_per_epoch:.4f} | "
            f"Pseudo={pseudo_sum/steps_per_epoch:.4f} (AccRate={accepted_frac_sum/steps_per_epoch*100:.1f}%) | "
            f"PV_Val Acc={val_acc * 100:.2f}% | PV_Val F1={val_f1:.4f} | Time={time.time() - t0:.1f}s"
        )

    del model, optimizer, scaler, scheduler, source_loader, target_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return save_path


def parse_args():
    parser = argparse.ArgumentParser(description="CropGuard v3 - Phase 5 Cross-Domain Transfer Trainer")
    parser.add_argument("--subset", type=str, default="all", choices=["pv_200", "pv_100", "all"])
    parser.add_argument("--stage", type=str, default="all", choices=["a", "b", "c", "all"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 quick forward/backward verification batch per stage")
    parser.add_argument("--seed", type=int, default=47)
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True


def run_dry_check(manifest: Dict[str, Any], device: torch.device) -> None:
    """Runs a quick 1-batch sanity check of Stage A, Stage B, and low-VRAM Stage C on CUDA/CPU."""
    logger.info("=== RUNNING PHASE 5 DRY-RUN SANITY CHECK ===")
    classes = manifest["classes"]
    sample_records = manifest["splits"]["pv_200"][:8]
    model = DinoV2LoRAPathologyModel(num_classes=len(classes)).to(device)
    ema_params = {n: p.data.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    scaler = torch.amp.GradScaler(device="cuda", enabled=(device.type == "cuda"))

    # Test Stage A & B datasets + forward/backward
    for st, rand_bg in [("a", False), ("b", True)]:
        ds = PVManifestDataset(sample_records, transform=get_source_transforms(st), randomize_bg=rand_bg)
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        x, y = next(iter(loader))
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits, _ = forward_with_features(model, x)
            loss = F.cross_entropy(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        logger.info(f"  [Dry-Run Stage {st.upper()}] Loss={float(loss.item()):.4f} | Logits={tuple(logits.shape)} OK")

    # Test Low-VRAM sequential Stage C UDA step (CORAL + SHOT IM + lightweight EMA swap)
    weak_tf, strong_tf = get_target_weak_strong_transforms()
    tgt_ds = UnlabeledTargetFieldDataset(PD_TRAIN_DIR, weak_tf, strong_tf)
    tw_list, ts_list = zip(*[tgt_ds[i] for i in range(4)])
    tw = torch.stack(tw_list).to(device)
    ts = torch.stack(ts_list).to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        backup = swap_trainable_params(model, ema_params)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            ema_logits, _ = model(tw, return_attention=False)
        swap_trainable_params(model, backup)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            _, feat_t_ref = forward_with_features(model, ts)
        feat_t_ref = feat_t_ref.detach()
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        logits_s, feat_s = forward_with_features(model, x)
        loss_s = F.cross_entropy(logits_s, y) + 0.5 * deep_coral_loss(feat_s, feat_t_ref)
    feat_s_ref = feat_s.detach()
    scaler.scale(loss_s).backward()
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        logits_t, feat_t = forward_with_features(model, ts)
        loss_c = deep_coral_loss(feat_s_ref, feat_t)
        loss_im = shot_information_maximization_loss(logits_t)
        loss_t = 0.5 * loss_c + 0.2 * loss_im
    scaler.scale(loss_t).backward()
    scaler.step(optimizer)
    scaler.update()
    update_ema_dict(ema_params, model)
    logger.info(f"  [Dry-Run Stage C] CORAL={float(loss_c.item()):.4f} | IM={float(loss_im.item()):.4f} | Low-VRAM EMA OK")
    logger.info("=== PHASE 5 DRY-RUN PASSED ===")


def main():
    args = parse_args()
    set_seed(args.seed)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}. Run prepare_pv_subsets.py first.")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    classes = manifest["classes"]
    val_records = manifest["splits"]["pv_val"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running Phase 5 Cross-Domain Training on {device}")

    if args.dry_run:
        run_dry_check(manifest, device)
        return

    val_ds = PVManifestDataset(val_records, transform=get_val_transforms(), randomize_bg=False)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    subsets = ["pv_200", "pv_100"] if args.subset == "all" else [args.subset]
    stages = ["a", "b", "c"] if args.stage == "all" else [args.stage]

    for subset_name in subsets:
        train_records = manifest["splits"][subset_name]
        epoch_map = {
            "pv_200": {"a": 5, "b": 8, "c": 5},
            "pv_100": {"a": 6, "b": 10, "c": 5},
        }[subset_name]

        stage_b_path = CKPT_DIR / f"{subset_name}_stage_b_best.pt"
        for st in stages:
            target_ckpt = CKPT_DIR / f"{subset_name}_stage_{st}_best.pt"
            if args.skip_existing and target_ckpt.exists():
                logger.info(f"Skipping existing checkpoint: {target_ckpt}")
                continue

            if st in ("a", "b"):
                p = train_source_stage(
                    subset_name=subset_name,
                    stage=st,
                    train_records=train_records,
                    val_loader=val_loader,
                    classes=classes,
                    epochs=epoch_map[st],
                    lr_max=2e-4,
                    batch_size=args.batch_size,
                    grad_accum_steps=args.grad_accum_steps,
                    num_workers=args.num_workers,
                    device=device,
                )
                if st == "b":
                    stage_b_path = p
            elif st == "c":
                if not stage_b_path.exists():
                    raise FileNotFoundError(f"Stage B checkpoint required before Stage C: {stage_b_path}")
                train_uda_stage_c(
                    subset_name=subset_name,
                    stage_b_ckpt_path=stage_b_path,
                    train_records=train_records,
                    val_loader=val_loader,
                    classes=classes,
                    epochs=epoch_map["c"],
                    lr_max=8e-5,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    device=device,
                )

    logger.info("Phase 5 cross-domain training pipeline complete.")


if __name__ == "__main__":
    main()
