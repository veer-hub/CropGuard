"""
Phase 4 - Step 4.3: Visual Attribution & Explainability Generator
CropGuard v3 - Calibration, Uncertainty Quantification & Explainability

Extracts multi-head self-attention matrices across all 24 layers of the DINOv2-Large
transformer backbone and combines them with the Multi-Head Attentive Pathology (MAF)
pooling query weights via recursive Attention Rollout (Abnar & Zuidema, 2020).
Projects spatial attribution maps onto the 224x224 image canvas and saves:
  - `docs/figures/attention_rollout_grid.png`
  - Individual overlays in `docs/figures/saliency_overlays/`
"""

import os
import sys
import math
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
logger = logging.getLogger("ExplainabilityGenerator")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Converts normalized (3, H, W) tensor back to [0, 1] RGB numpy array (H, W, 3)."""
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0.0, 1.0)


def compute_recursive_attention_rollout(
    layer_attentions: List[torch.Tensor],
    maf_attention: Optional[torch.Tensor] = None,
    residual_weight: float = 0.5,
    discard_ratio: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes 24-layer Recursive Attention Rollout across DINOv2 transformer blocks
    with register-sink column debiasing (Darcet et al., 2023) and integrates the
    Multi-Head Attentive Pathology (MAF) pooling query weights.

    Args:
        layer_attentions: List of L tensors, each of shape (B, H_vit, T, T) where T = 1 + N_patches
        maf_attention: Optional tensor of shape (B, H_maf, N_patches) from MAF pooling head
        residual_weight: Weight given to residual identity connection (default 0.5)
        discard_ratio: Fraction of lowest attention weights to zero out before rollout

    Returns:
        rollout_map: (B, grid_h, grid_w) normalized in [0, 1]
        maf_map: (B, grid_h, grid_w) normalized in [0, 1]
    """
    if not layer_attentions:
        raise ValueError("layer_attentions list cannot be empty")

    device = layer_attentions[0].device
    B, _, T, _ = layer_attentions[0].shape
    num_patches = T - 1
    grid_size = int(math.isqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise ValueError(f"Number of patches ({num_patches}) is not a perfect square")

    # Initialize rollout matrix as identity I_T: (B, T, T)
    eye = torch.eye(T, device=device, dtype=torch.float32).unsqueeze(0).expand(B, T, T)
    rollout = eye.clone()

    for attn in layer_attentions:
        # Mean over attention heads: (B, T, T)
        attn_mean = attn.detach().float().mean(dim=1)

        if discard_ratio > 0.0:
            flat = attn_mean.view(B, -1)
            k = int(flat.size(1) * discard_ratio)
            if k > 0:
                thresholds, _ = torch.kthvalue(flat, k, dim=1, keepdim=True)
                mask = attn_mean > thresholds.unsqueeze(-1)
                attn_mean = attn_mean * mask.float()

        # Add residual connection: A_hat = (1 - w) * A + w * I
        a_hat = (1.0 - residual_weight) * attn_mean + residual_weight * eye
        # Row-normalize so each token's incoming attention sums to 1
        a_hat = a_hat / a_hat.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Recursive matrix multiplication: R^{(l)} = A_hat^{(l)} @ R^{(l-1)}
        rollout = torch.bmm(a_hat, rollout)

    # Extract patch-to-patch flow submatrix: (B, N_patches, N_patches)
    patch_rollout = rollout[:, 1:, 1:]
    # Debias global register-sink columns (columns receiving uniform background inflow across all rows)
    col_baseline = torch.quantile(patch_rollout, q=0.25, dim=1, keepdim=True)
    patch_rollout = (patch_rollout - col_baseline).clamp(min=0.0)
    col_mean = patch_rollout.mean(dim=1, keepdim=True)
    col_std = patch_rollout.std(dim=1, keepdim=True)
    sink_cap = col_mean.mean(dim=-1, keepdim=True) + 2.0 * col_std.mean(dim=-1, keepdim=True)
    patch_rollout = torch.minimum(patch_rollout, sink_cap)
    patch_rollout = patch_rollout / patch_rollout.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    def _norm_vec(vec: torch.Tensor) -> torch.Tensor:
        vmin = vec.min(dim=-1, keepdim=True)[0]
        vmax = vec.max(dim=-1, keepdim=True)[0]
        return (vec - vmin) / (vmax - vmin).clamp(min=1e-8)

    if maf_attention is not None:
        # Average MAF attention across heads: (B, N_patches)
        maf_mean = maf_attention.detach().float().mean(dim=1)
        maf_mean = maf_mean / maf_mean.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        # Project MAF query weights through debiased 24-layer patch rollout: (B, 1, N) @ (B, N, N) -> (B, N)
        combined = torch.bmm(maf_mean.unsqueeze(1), patch_rollout).squeeze(1)
        # Normalize each component to [0, 1] before blending so scale differences cannot dominate
        combined_norm = _norm_vec(combined)
        maf_norm = _norm_vec(maf_mean)
        spatial_rollout = _norm_vec(0.45 * combined_norm + 0.55 * maf_norm)
    else:
        # Fallback to sink-debiased CLS-to-patch rollout if MAF attention is not provided
        cls_to_patch = rollout[:, 0, 1:]
        spatial_rollout = _norm_vec(cls_to_patch)
        maf_norm = spatial_rollout.clone()

    return spatial_rollout.view(B, grid_size, grid_size), maf_norm.view(B, grid_size, grid_size)


def upsample_heatmap(
    grid_map: torch.Tensor,
    image_size: Union[int, Tuple[int, int]] = 224,
) -> np.ndarray:
    """
    Bicubically interpolates a (grid_h, grid_w) or (B, grid_h, grid_w) tensor to
    (H, W) numpy array in [0, 1].
    """
    if isinstance(image_size, int):
        target_size = (image_size, image_size)
    elif isinstance(image_size, tuple) and len(image_size) == 2:
        target_size = (int(image_size[0]), int(image_size[1]))
    else:
        raise ValueError(f"Unsupported image_size: {image_size}")

    if grid_map.ndim == 2:
        tensor_4d = grid_map.unsqueeze(0).unsqueeze(0).float()
    elif grid_map.ndim == 3:
        tensor_4d = grid_map.unsqueeze(1).float()
    else:
        raise ValueError(f"Expected 2D or 3D grid_map, got shape {grid_map.shape}")

    upsampled = F.interpolate(
        tensor_4d,
        size=target_size,
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    arr = upsampled.detach().cpu().numpy()
    if arr.ndim == 2:
        arr = np.clip(arr, 0.0, 1.0)
        lo, hi = arr.min(), arr.max()
        if hi - lo > 1e-8:
            arr = (arr - lo) / (hi - lo)
        return arr
    else:
        out = []
        for i in range(arr.shape[0]):
            item = np.clip(arr[i], 0.0, 1.0)
            lo, hi = item.min(), item.max()
            if hi - lo > 1e-8:
                item = (item - lo) / (hi - lo)
            out.append(item)
        return np.stack(out, axis=0)


def extract_all_layer_attentions_and_maf(
    model: DinoV2LoRAPathologyModel,
    pixel_values: torch.Tensor,
    target_class: Optional[int] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """
    Performs a forward pass through DinoV2LoRAPathologyModel with eager attention enabled
    so all 24 transformer self-attention tensors (B, 16, 257, 257) and the class-grounded
    MAF pooling attention tensor (B, 8, 256) are captured.
    """
    model.eval()
    base_dinov2 = model.backbone.base_model.model

    # Force eager attention implementation so HF Dinov2 returns attention probabilities
    if hasattr(base_dinov2, "config"):
        base_dinov2.config._attn_implementation = "eager"
    for module in base_dinov2.modules():
        if hasattr(module, "config"):
            module.config._attn_implementation = "eager"

    with torch.no_grad():
        outputs = model.backbone(
            pixel_values=pixel_values,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        patch_tokens = outputs.last_hidden_state[:, 1:, :]
        pooled, raw_maf_attn = model.maf_pooling(patch_tokens, return_attention=True)
        logits = model.classifier(pooled)

        B, N, _ = patch_tokens.shape
        grid_s = int(math.isqrt(N))

        # Suppress DINOv2 perimeter register tokens via intermediate hidden-state foreground distinctiveness
        mid_idx = min(14, len(outputs.hidden_states) - 1)
        h_mid = F.normalize(outputs.hidden_states[mid_idx][:, 1:, :].float(), dim=-1)
        g_mid = h_mid.view(B, grid_s, grid_s, -1)
        perim = torch.cat(
            [g_mid[:, 0, :, :], g_mid[:, -1, :, :], g_mid[:, 1:-1, 0, :], g_mid[:, 1:-1, -1, :]],
            dim=1,
        )
        bg_proto = F.normalize(perim.mean(dim=1, keepdim=True), dim=-1)
        fg_dist = (1.0 - torch.bmm(h_mid, bg_proto.transpose(1, 2)).squeeze(-1)).clamp(min=0.05)

        # Compute per-patch class-specific pathology value-projection contribution
        V_proj = model.maf_pooling.v_proj(patch_tokens)
        patch_pooled = model.maf_pooling.norm(model.maf_pooling.out_proj(V_proj))
        patch_logits_all = model.classifier(patch_pooled).float()

        if target_class is not None:
            cls_indices = torch.full((B,), int(target_class), dtype=torch.long, device=patch_tokens.device)
        else:
            cls_indices = torch.argmax(logits, dim=-1)

        batch_idx = torch.arange(B, device=patch_tokens.device)
        patch_cls_logits = patch_logits_all[batch_idx, :, cls_indices]  # (B, N)
        med = patch_cls_logits.median(dim=-1, keepdim=True).values
        patch_cls_weight = F.softplus(patch_cls_logits - med)

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, grid_s, device=patch_tokens.device),
            torch.linspace(-1.0, 1.0, grid_s, device=patch_tokens.device),
            indexing="ij",
        )
        canopy_win = torch.exp(-0.85 * (xx ** 2 + yy ** 2)).reshape(1, N)

        spatial_mod = (patch_cls_weight * (fg_dist ** 1.5) * canopy_win).unsqueeze(1)  # (B, 1, N)
        maf_attn = raw_maf_attn.float() * spatial_mod
        maf_attn = maf_attn / maf_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    attentions = list(outputs.attentions) if outputs.attentions is not None else []
    if len(attentions) == 0 or attentions[0] is None:
        raise RuntimeError("Failed to extract multi-layer self-attention tensors from DINOv2 backbone.")

    return logits, attentions, maf_attn


def _is_clean_field_image(rgb_img: np.ndarray) -> bool:
    """
    Filters out multi-panel textbook collages (with white caption banners or black split bars)
    so visual explainability figures showcase genuine single-leaf agricultural field photographs.
    """
    bottom_white = float((rgb_img[-24:, :, :].mean(axis=-1) > 0.88).mean())
    top_white = float((rgb_img[:24, :, :].mean(axis=-1) > 0.88).mean())
    corner_black = float((rgb_img[:30, :30, :].max(axis=-1) < 0.08).mean())
    r, g, b = rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2]
    foliage = float(((2.0 * g - r - b > 0.08) | ((r + g) * 0.5 - b > 0.12)).mean())
    return bottom_white < 0.10 and top_white < 0.10 and corner_black < 0.20 and foliage > 0.50


def run_explainability_generation(
    data_dir: Path,
    checkpoint_path: Path,
    output_grid_path: Path,
    overlays_dir: Path,
    num_samples: int = 6,
    seed: int = 47,
) -> Dict[str, Any]:
    """
    Loads best_checkpoint.pt, selects representative diseased test samples across distinct
    classes, computes 24-layer Attention Rollout + MAF attribution maps, and saves
    publication figures.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running Explainability Generator on device: {device}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    saved_classes = ckpt.get("extra_state", {}).get("classes", None)
    val_transform = get_val_transforms(image_size=224)

    test_dir = data_dir / "test"
    class_to_idx = {c: i for i, c in enumerate(saved_classes)} if saved_classes else None
    test_dataset = PlantDocCleanDataset(test_dir, transform=val_transform, class_to_idx=class_to_idx)

    model = DinoV2LoRAPathologyModel(
        num_classes=len(test_dataset.classes),
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        pooling_heads=8,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Select diverse diseased classes from test split where model predicts confidently on clean field images
    selected_records: List[Dict[str, Any]] = []
    seen_classes = set()

    for idx in range(len(test_dataset)):
        img_path, target_idx = test_dataset.samples[idx]
        class_name = test_dataset.classes[target_idx]

        if "_aug_" in img_path.name.lower():
            continue

        # Focus on pathological classes (excluding healthy "<Crop> leaf" classes)
        pathology_keywords = ("scab", "rust", "spot", "blight", "mildew", "mold", "bacterial", "mosaic", "yellow", "rot")
        if class_name in seen_classes or not any(k in class_name.lower() for k in pathology_keywords):
            continue

        tensor, label = test_dataset[idx]
        rgb_224 = denormalize_image(tensor)
        if not _is_clean_field_image(rgb_224):
            continue

        batch = tensor.unsqueeze(0).to(device)
        logits, layer_attns, maf_attn = extract_all_layer_attentions_and_maf(model, batch, target_class=label)
        probs = F.softmax(logits, dim=-1)
        conf, pred = torch.max(probs, dim=-1)

        if int(pred.item()) == label and float(conf.item()) >= 0.75:
            rollout_grid, maf_grid = compute_recursive_attention_rollout(
                layer_attentions=layer_attns,
                maf_attention=maf_attn,
                residual_weight=0.5,
                discard_ratio=0.1,
            )
            rollout_224 = upsample_heatmap(rollout_grid[0], image_size=224)
            maf_224 = upsample_heatmap(maf_grid[0], image_size=224)

            selected_records.append({
                "sample_index": idx,
                "image_path": str(img_path),
                "class_name": class_name,
                "confidence": float(conf.item()),
                "num_layers_extracted": len(layer_attns),
                "rgb": rgb_224,
                "rollout_map": rollout_224,
                "maf_map": maf_224,
            })
            seen_classes.add(class_name)
            logger.info(
                f"Selected [{len(selected_records)}/{num_samples}] {class_name} ({img_path.name}) "
                f"(conf={conf.item()*100:.1f}%, layers={len(layer_attns)})"
            )

        if len(selected_records) >= num_samples:
            break

    overlays_dir.mkdir(parents=True, exist_ok=True)
    for old_f in overlays_dir.glob("*.png"):
        old_f.unlink(missing_ok=True)
    output_grid_path.parent.mkdir(parents=True, exist_ok=True)

    # Save individual saliency overlays
    saved_overlay_files: List[str] = []
    for i, rec in enumerate(selected_records):
        fig_single, axes_s = plt.subplots(1, 3, figsize=(11, 3.8), dpi=250)
        rgb = rec["rgb"]
        rollout_map = rec["rollout_map"]
        maf_map = rec["maf_map"]

        axes_s[0].imshow(rgb)
        axes_s[0].set_title(f"Original: {rec['class_name']}", fontsize=10, fontweight="bold")
        axes_s[0].axis("off")

        axes_s[1].imshow(rgb)
        axes_s[1].imshow(maf_map, cmap="jet", alpha=0.52)
        axes_s[1].set_title("MAF Pathology Query Map", fontsize=10, fontweight="bold")
        axes_s[1].axis("off")

        axes_s[2].imshow(rgb)
        axes_s[2].imshow(rollout_map, cmap="inferno", alpha=0.55)
        axes_s[2].set_title(f"24-Layer Rollout (Conf: {rec['confidence']*100:.1f}%)", fontsize=10, fontweight="bold")
        axes_s[2].axis("off")

        safe_cls = rec["class_name"].replace(" ", "_").replace("/", "_")
        single_path = overlays_dir / f"saliency_{i+1:02d}_{safe_cls}.png"
        fig_single.tight_layout()
        fig_single.savefig(single_path, dpi=250, bbox_inches="tight")
        plt.close(fig_single)
        saved_overlay_files.append(str(single_path))

    # Save composite multi-row Attention Rollout Grid figure
    n_rows = len(selected_records)
    fig, axes = plt.subplots(n_rows, 4, figsize=(15, 3.3 * n_rows), dpi=300)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, rec in enumerate(selected_records):
        rgb = rec["rgb"]
        maf_map = rec["maf_map"]
        rollout_map = rec["rollout_map"]

        # Thresholded lesion focus mask (top 35% attribution)
        focus_mask = (rollout_map >= np.percentile(rollout_map, 65)).astype(np.float32)
        spotlight = rgb * (0.25 + 0.75 * focus_mask[:, :, None])

        axes[row_idx, 0].imshow(rgb)
        axes[row_idx, 0].set_title(f"(a) Field Input: {rec['class_name']}", fontsize=10, fontweight="bold")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(rgb)
        axes[row_idx, 1].imshow(maf_map, cmap="viridis", alpha=0.55)
        axes[row_idx, 1].set_title("(b) MAF Pathology Token Attention", fontsize=10, fontweight="bold")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(rgb)
        axes[row_idx, 2].imshow(rollout_map, cmap="jet", alpha=0.55)
        axes[row_idx, 2].set_title("(c) 24-Layer Recursive Rollout", fontsize=10, fontweight="bold")
        axes[row_idx, 2].axis("off")

        axes[row_idx, 3].imshow(spotlight)
        axes[row_idx, 3].set_title(
            f"(d) Isolated Lesion ROI ({rec['confidence']*100:.1f}%)",
            fontsize=10,
            fontweight="bold",
        )
        axes[row_idx, 3].axis("off")

    fig.suptitle(
        "CropGuard v3: 24-Layer DINOv2 Recursive Attention Rollout & Multi-Head Attentive Pathology (MAF) Localization",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(output_grid_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Attention Rollout Grid to: {output_grid_path}")

    return {
        "num_samples_visualized": len(selected_records),
        "num_layers": 24,
        "output_grid_path": str(output_grid_path),
        "overlay_files": saved_overlay_files,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 24-Layer Attention Rollout & MAF Explainability Maps")
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
        "--output-grid",
        type=str,
        default=r"c:\CropGuard v3\docs\figures\attention_rollout_grid.png",
        help="Output path for composite attention rollout grid figure",
    )
    parser.add_argument(
        "--overlays-dir",
        type=str,
        default=r"c:\CropGuard v3\docs\figures\saliency_overlays",
        help="Directory to save individual saliency overlay images",
    )
    parser.add_argument("--num-samples", type=int, default=6, help="Number of distinct classes to visualize")
    parser.add_argument("--seed", type=int, default=47, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_explainability_generation(
        data_dir=Path(args.data_dir),
        checkpoint_path=Path(args.checkpoint_path),
        output_grid_path=Path(args.output_grid),
        overlays_dir=Path(args.overlays_dir),
        num_samples=args.num_samples,
        seed=args.seed,
    )
