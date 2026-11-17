"""
Phase 5 - Step 5.2b & 5.3: Cross-Domain Zero-Shot Evaluation & Publication Plotter
CropGuard v3 - PlantVillage (PV_200 & PV_100) -> PlantDoc Zero-Shot Transfer

Evaluates Stage A (Source-Only Lab), Stage B (Segmented Background Randomization + MAF),
and Stage C (Full CropGuard v3 Domain-Adaptive Transfer) on the strictly held-out,
zero-leakage PlantDoc test set (`augmented_plantdoc_clean/test`, N=1,588, 27 classes).

Computes Top-1 Accuracy, Macro-F1, Macro-Precision, Macro-Recall, Per-Class F1,
Post-Hoc Calibration (ECE via pv_val temperature scaling), caches logits, exports
`docs/cross_domain_results.json`, and generates `docs/figures/cross_domain_comparison.png`.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_dinov2_lora import (
    DinoV2LoRAPathologyModel,
    PlantDocCleanDataset,
    get_val_transforms,
)
from train_cross_domain_pv import PVManifestDataset
from evaluate_calibration import compute_calibration_metrics, optimize_temperature_lbfgs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("CrossDomainEval")

BASE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = BASE_DIR / "docs" / "pv_subsets_manifest.json"
PD_TEST_DIR = BASE_DIR / "augmented_plantdoc_clean" / "test"
CKPT_DIR = BASE_DIR / "checkpoints" / "cross_domain_pv"
RESULTS_PATH = BASE_DIR / "docs" / "cross_domain_results.json"
LOGITS_CACHE_PATH = CKPT_DIR / "cross_domain_logits_cache.pt"
FIG_PATH = BASE_DIR / "docs" / "figures" / "cross_domain_comparison.png"

SALMAN_TABLE_4_PV200 = [
    ("API-Net", 0.39, 0.38, 0.39, 0.39),
    ("InceptionV3", 0.41, 0.40, 0.40, 0.40),
    ("EfficientNet", 0.43, 0.42, 0.42, 0.43),
    ("CAL", 0.44, 0.44, 0.44, 0.44),
    ("TransFG", 0.44, 0.42, 0.42, 0.43),
    ("ViT-Base", 0.48, 0.47, 0.46, 0.48),
    ("Ensemble CNN", 0.51, 0.50, 0.50, 0.50),
    ("MoE (No Reg)", 0.60, 0.57, 0.56, 0.60),
    ("Salman et al. (ViT-MoE)", 0.68, 0.65, 0.67, 0.64),
]

SALMAN_TABLE_5_PV100 = [
    ("EfficientNet", 0.28, 0.28, 0.19, 0.23),
    ("InceptionV3", 0.29, 0.29, 0.28, 0.29),
    ("ViT-Base", 0.30, 0.35, 0.44, 0.30),
    ("API-Net", 0.32, 0.32, 0.32, 0.32),
    ("TransFG", 0.37, 0.36, 0.37, 0.35),
    ("CAL", 0.37, 0.37, 0.37, 0.37),
    ("Ensemble CNN", 0.41, 0.40, 0.44, 0.36),
    ("Salman et al. (ViT-MoE)", 0.49, 0.51, 0.53, 0.49),
]


@torch.no_grad()
def extract_logits_for_loader(
    model: DinoV2LoRAPathologyModel,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts float32 logits and ground-truth labels for an evaluation loader."""
    model.eval()
    logits_list = []
    targets_list = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits, _ = model(images, return_attention=False)
        logits_list.append(logits.float().cpu())
        targets_list.append(targets.long().cpu())
    return torch.cat(logits_list, dim=0), torch.cat(targets_list, dim=0)


def evaluate_single_checkpoint(
    ckpt_path: Path,
    model: DinoV2LoRAPathologyModel,
    pv_val_loader: DataLoader,
    pd_test_loader: DataLoader,
    classes: List[str],
    device: torch.device,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """Evaluates a single cross-domain checkpoint zero-shot on PlantDoc Test."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    val_logits, val_targets = extract_logits_for_loader(model, pv_val_loader, device)
    test_logits, test_targets = extract_logits_for_loader(model, pd_test_loader, device)

    # Optimize scalar temperature strictly on PV_Val (zero target tuning)
    t_star = optimize_temperature_lbfgs(val_logits, val_targets)
    cal_uncal = compute_calibration_metrics(test_logits, test_targets, temperature=1.0, num_bins=15)
    cal_scaled = compute_calibration_metrics(test_logits, test_targets, temperature=t_star, num_bins=15)

    y_true = test_targets.numpy()
    y_pred = torch.argmax(test_logits, dim=-1).numpy()

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cls_rep = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    per_class_f1 = {c: round(float(cls_rep[c]["f1-score"]), 4) for c in classes}

    metrics = {
        "checkpoint": str(ckpt_path),
        "epoch": int(ckpt.get("epoch", -1)),
        "pv_val_accuracy": round(float(ckpt.get("pv_val_acc", 0.0)), 6),
        "pv_val_macro_f1": round(float(ckpt.get("pv_val_macro_f1", 0.0)), 6),
        "plantdoc_test_samples": int(len(y_true)),
        "plantdoc_test_accuracy": round(acc, 6),
        "plantdoc_test_macro_f1": round(macro_f1, 6),
        "plantdoc_test_macro_precision": round(macro_prec, 6),
        "plantdoc_test_macro_recall": round(macro_rec, 6),
        "plantdoc_test_weighted_f1": round(weighted_f1, 6),
        "pv_val_scalar_temperature": round(float(t_star), 6),
        "plantdoc_test_ece_uncalibrated": round(float(cal_uncal["ece"]), 6),
        "plantdoc_test_ece_calibrated": round(float(cal_scaled["ece"]), 6),
        "per_class_f1": per_class_f1,
    }
    cache = {
        "val_logits": val_logits,
        "val_targets": val_targets,
        "test_logits": test_logits,
        "test_targets": test_targets,
        "t_star": float(t_star),
    }
    return metrics, cache


def plot_cross_domain_comparison(results: Dict[str, Any], out_path: Path) -> None:
    """Generates multi-panel publication figure comparing CropGuard v3 against Salman et al. Tables 4 & 5."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#444444"})
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.2), dpi=300)

    # Panel A: PV_200 -> PlantDoc Benchmark (Salman Table 4 vs CropGuard v3)
    ax_a = axes[0]
    pv200_c_acc = results["subsets"]["pv_200"]["stages"]["c"]["plantdoc_test_accuracy"] * 100.0
    models_200 = [m for m, *_ in SALMAN_TABLE_4_PV200] + ["CropGuard v3\n(Ours, Raw PD)"]
    accs_200 = [a * 100.0 for _, a, *_ in SALMAN_TABLE_4_PV200] + [pv200_c_acc]
    colors_200 = ["#90A4AE"] * (len(SALMAN_TABLE_4_PV200) - 1) + ["#F57C00", "#1B5E20"]

    bars_a = ax_a.barh(models_200, accs_200, color=colors_200, edgecolor="#263238", linewidth=0.7)
    ax_a.axvline(68.0, color="#F57C00", linestyle="--", linewidth=1.2, alpha=0.8)
    ax_a.set_xlim(0, 95)
    ax_a.set_xlabel("PlantDoc Zero-Shot Accuracy (%)")
    ax_a.set_title("A. $PV_{200} \\rightarrow$ PlantDoc (Table 4 Comparison)", fontweight="bold", pad=10)
    ax_a.grid(axis="x", linestyle=":", alpha=0.5)
    for bar, val in zip(bars_a, accs_200):
        ax_a.text(val + 1.2, bar.get_y() + bar.get_height() / 2, f"{val:.2f}%", va="center", fontsize=8.5, fontweight="bold")

    # Panel B: PV_100 -> PlantDoc Benchmark (Salman Table 5 vs CropGuard v3)
    ax_b = axes[1]
    pv100_c_acc = results["subsets"]["pv_100"]["stages"]["c"]["plantdoc_test_accuracy"] * 100.0
    models_100 = [m for m, *_ in SALMAN_TABLE_5_PV100] + ["CropGuard v3\n(Ours, Raw PD)"]
    accs_100 = [a * 100.0 for _, a, *_ in SALMAN_TABLE_5_PV100] + [pv100_c_acc]
    colors_100 = ["#90A4AE"] * (len(SALMAN_TABLE_5_PV100) - 1) + ["#F57C00", "#1B5E20"]

    bars_b = ax_b.barh(models_100, accs_100, color=colors_100, edgecolor="#263238", linewidth=0.7)
    ax_b.axvline(49.0, color="#F57C00", linestyle="--", linewidth=1.2, alpha=0.8)
    ax_b.set_xlim(0, 95)
    ax_b.set_xlabel("PlantDoc Zero-Shot Accuracy (%)")
    ax_b.set_title("B. $PV_{100} \\rightarrow$ PlantDoc (Table 5 Comparison)", fontweight="bold", pad=10)
    ax_b.grid(axis="x", linestyle=":", alpha=0.5)
    for bar, val in zip(bars_b, accs_100):
        ax_b.text(val + 1.2, bar.get_y() + bar.get_height() / 2, f"{val:.2f}%", va="center", fontsize=8.5, fontweight="bold")

    # Panel C: 3-Stage Ablation Progression (Stage A -> Stage B -> Stage C)
    ax_c = axes[2]
    stage_labels = ["Stage A\n(Lab Only ERM)", "Stage B\n(Seg BG Rand\n+ MAF)", "Stage C\n(Full Domain\nAdaptive)"]
    pv100_vals = [
        results["subsets"]["pv_100"]["stages"][s]["plantdoc_test_accuracy"] * 100.0
        for s in ("a", "b", "c")
    ]
    pv200_vals = [
        results["subsets"]["pv_200"]["stages"][s]["plantdoc_test_accuracy"] * 100.0
        for s in ("a", "b", "c")
    ]
    x = np.arange(len(stage_labels))
    width = 0.34
    rects1 = ax_c.bar(x - width / 2, pv100_vals, width, label="$PV_{100}$ (100/cls)", color="#43A047", edgecolor="#263238")
    rects2 = ax_c.bar(x + width / 2, pv200_vals, width, label="$PV_{200}$ (200/cls)", color="#1B5E20", edgecolor="#263238")
    ax_c.axhline(68.0, color="#F57C00", linestyle="--", linewidth=1.2, label="Salman $PV_{200}$ (68%)")
    ax_c.axhline(49.0, color="#FB8C00", linestyle=":", linewidth=1.2, label="Salman $PV_{100}$ (49%)")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(stage_labels, fontsize=9)
    ax_c.set_ylim(0, 95)
    ax_c.set_ylabel("PlantDoc Zero-Shot Accuracy (%)")
    ax_c.set_title("C. Domain Generalization Ablation", fontweight="bold", pad=10)
    ax_c.legend(loc="upper left", fontsize=8.5)
    ax_c.grid(axis="y", linestyle=":", alpha=0.5)

    for rect in list(rects1) + list(rects2):
        h = rect.get_height()
        ax_c.text(rect.get_x() + rect.get_width() / 2.0, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    fig.suptitle(
        "CropGuard v3 Cross-Domain Transfer ($PlantVillage \\rightarrow$ Un-Pruned Zero-Leakage PlantDoc, $N=1,588$)",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    tmp_fig = out_path.with_suffix(".png.tmp")
    fig.savefig(tmp_fig, dpi=300, format="png")
    plt.close(fig)
    os.replace(tmp_fig, out_path)
    logger.info(f"Saved publication comparison figure to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="CropGuard v3 - Phase 5 Cross-Domain Zero-Shot Evaluation")
    parser.add_argument("--subset", type=str, default="all", choices=["pv_200", "pv_100", "all"])
    return parser.parse_args()


def main():
    args = parse_args()
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest missing: {MANIFEST_PATH}")
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    classes = manifest["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    pd_test_ds = PlantDocCleanDataset(PD_TEST_DIR, transform=get_val_transforms(), class_to_idx=class_to_idx)
    assert pd_test_ds.classes == classes, "Class alignment mismatch between PlantVillage manifest and PlantDoc test!"

    pv_val_ds = PVManifestDataset(manifest["splits"]["pv_val"], transform=get_val_transforms(), randomize_bg=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pv_val_loader = DataLoader(pv_val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))
    pd_test_loader = DataLoader(pd_test_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))

    model = DinoV2LoRAPathologyModel(num_classes=len(classes)).to(device)

    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = {
            "model": "DINOv2-Large + LoRA (r=16, alpha=32) + MAF Pooling",
            "target_dataset": str(PD_TEST_DIR),
            "target_num_samples": len(pd_test_ds),
            "num_classes": len(classes),
            "salman_2025_benchmarks": {
                "pv_200_accuracy": 0.68,
                "pv_200_f1": 0.65,
                "pv_100_accuracy": 0.49,
                "pv_100_f1": 0.51,
            },
            "subsets": {},
        }

    logits_cache: Dict[str, Any] = torch.load(LOGITS_CACHE_PATH, map_location="cpu") if LOGITS_CACHE_PATH.exists() else {}

    target_subsets = ("pv_200", "pv_100") if args.subset == "all" else (args.subset,)
    for subset_name in target_subsets:
        subset_res: Dict[str, Any] = {"stages": {}}
        logits_cache[subset_name] = {}
        for st in ("a", "b", "c"):
            ckpt_path = CKPT_DIR / f"{subset_name}_stage_{st}_best.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Missing checkpoint for {subset_name} stage {st}: {ckpt_path}")
            logger.info(f"Evaluating {subset_name.upper()} Stage {st.upper()} zero-shot on PlantDoc Test (N={len(pd_test_ds)})...")
            m, cache = evaluate_single_checkpoint(ckpt_path, model, pv_val_loader, pd_test_loader, classes, device)
            subset_res["stages"][st] = m
            logits_cache[subset_name][st] = cache
            logger.info(
                f"  -> [{subset_name.upper()} Stage {st.upper()}] Test Acc={m['plantdoc_test_accuracy']*100:.2f}% | "
                f"Macro-F1={m['plantdoc_test_macro_f1']:.4f} | Prec={m['plantdoc_test_macro_precision']:.4f} | "
                f"Rec={m['plantdoc_test_macro_recall']:.4f} | ECE={m['plantdoc_test_ece_calibrated']:.4f}"
            )
        results["subsets"][subset_name] = subset_res

    # Atomic save logits cache and JSON report
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_cache = LOGITS_CACHE_PATH.with_suffix(".pt.tmp")
    with open(tmp_cache, "wb") as f:
        torch.save(logits_cache, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_cache, LOGITS_CACHE_PATH)
    logger.info(f"Saved cross-domain logits cache to {LOGITS_CACHE_PATH}")

    tmp_res = RESULTS_PATH.with_suffix(".json.tmp")
    with open(tmp_res, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_res, RESULTS_PATH)
    logger.info(f"Saved cross-domain evaluation results to {RESULTS_PATH}")

    if "pv_200" in results["subsets"] and "pv_100" in results["subsets"]:
        plot_cross_domain_comparison(results, FIG_PATH)
    else:
        logger.info("Skipping 3-panel comparison plot until both pv_200 and pv_100 are evaluated.")


if __name__ == "__main__":
    main()
