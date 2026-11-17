"""
Phase 5 - Step 5.1: PlantVillage Cross-Domain Subset Builder (PV_200, PV_100, PV_Val)
CropGuard v3 - Cross-Domain Lab-to-Field Transfer (PlantVillage -> PlantDoc)

Constructs strictly balanced, diversity-sampled, zero-leakage PlantVillage subsets
across the 27 PlantDoc-aligned classes:
  - PV_200: 200 diverse training images per class (5,400 total)
  - PV_100: 100 diverse training images per class (2,700 total, strict subset of PV_200)
  - PV_Val: 50 disjoint validation images per class (1,350 total, zero overlap with PV_200)

Also maps every color image to its corresponding cutout in `plantvillage dataset/segmented`
and records RGB channel distribution statistics across PlantVillage vs PlantDoc.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("PVSubsetBuilder")

BASE_DIR = Path(__file__).resolve().parent
PV_COLOR_DIR = BASE_DIR / "plantvillage dataset" / "color"
PV_SEG_DIR = BASE_DIR / "plantvillage dataset" / "segmented"
PD_CLEAN_TEST_DIR = BASE_DIR / "augmented_plantdoc_clean" / "test"
MANIFEST_PATH = BASE_DIR / "docs" / "pv_subsets_manifest.json"
RANDOM_SEED = 47

# Exact 1-to-1 class mapping between the 27 PlantDoc classes and PlantVillage classes
PD_TO_PV_CLASS_MAP: Dict[str, str] = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
}


def find_segmented_match(seg_class_dir: Path, color_path: Path, seg_lookup: Dict[str, Path]) -> Optional[str]:
    """Finds the matching segmented cutout for a given PlantVillage color image."""
    stem = color_path.stem
    candidate_exact = f"{stem}_final_masked".lower()
    if candidate_exact in seg_lookup:
        return str(seg_lookup[candidate_exact])
    if stem.lower() in seg_lookup:
        return str(seg_lookup[stem.lower()])
    # Fallback UUID prefix before '___'
    prefix = stem.split("___")[0].lower()
    if prefix in seg_lookup:
        return str(seg_lookup[prefix])
    # Fallback suffix after '___' (handles Corn Common rust where color has no UUID prefix)
    suffix = stem.split("___")[-1].lower()
    if suffix in seg_lookup:
        return str(seg_lookup[suffix])
    return None


def extract_lightweight_descriptor(img_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts a lightweight 198-D spatial+color descriptor and RGB mean/std."""
    with Image.open(img_path) as img:
        rgb = img.convert("RGB")
        arr_224 = np.asarray(rgb.resize((224, 224), Image.BILINEAR), dtype=np.float32) / 255.0
        rgb_mean = arr_224.mean(axis=(0, 1))
        rgb_std = arr_224.std(axis=(0, 1))
        thumb = np.asarray(rgb.resize((8, 8), Image.BILINEAR), dtype=np.float32) / 255.0
        feat = np.concatenate([thumb.flatten(), rgb_mean, rgb_std], axis=0)
    return feat, rgb_mean, rgb_std


def select_diverse_medoids(features: np.ndarray, k: int, seed: int) -> List[int]:
    """Selects k diverse representative indices using MiniBatchKMeans cluster medoids."""
    n = features.shape[0]
    if k >= n:
        return list(range(n))
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=min(256, n))
    labels = km.fit_predict(features)
    centers = km.cluster_centers_

    selected: List[int] = []
    used = set()
    for cid in range(k):
        members = np.where(labels == cid)[0]
        if len(members) > 0:
            dists = np.linalg.norm(features[members] - centers[cid], axis=1)
            order = np.argsort(dists)
            chosen = None
            for idx in order:
                cand = int(members[idx])
                if cand not in used:
                    chosen = cand
                    break
            if chosen is not None:
                selected.append(chosen)
                used.add(chosen)

    # Top up deterministically if any cluster was empty
    if len(selected) < k:
        rng = np.random.default_rng(seed)
        remaining = [i for i in range(n) if i not in used]
        rng.shuffle(remaining)
        needed = k - len(selected)
        selected.extend(remaining[:needed])

    return selected


def build_pv_subsets(
    n_train_200: int = 200,
    n_train_100: int = 100,
    n_val: int = 50,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Builds strictly disjoint PV_200, PV_100, and PV_Val subsets across all 27 classes."""
    if not PV_COLOR_DIR.exists():
        raise FileNotFoundError(f"PlantVillage color directory not found: {PV_COLOR_DIR}")
    if not PV_SEG_DIR.exists():
        raise FileNotFoundError(f"PlantVillage segmented directory not found: {PV_SEG_DIR}")

    sorted_pd_classes = sorted(PD_TO_PV_CLASS_MAP.keys())
    logger.info(f"Building PlantVillage subsets for {len(sorted_pd_classes)} aligned classes...")

    pv_200_records: List[Dict[str, Any]] = []
    pv_100_records: List[Dict[str, Any]] = []
    pv_val_records: List[Dict[str, Any]] = []
    class_summary: Dict[str, Dict[str, Any]] = {}

    pv_means = []
    pv_stds = []

    for cidx, pd_class in enumerate(sorted_pd_classes):
        pv_class = PD_TO_PV_CLASS_MAP[pd_class]
        color_cdir = PV_COLOR_DIR / pv_class
        seg_cdir = PV_SEG_DIR / pv_class

        if not color_cdir.exists():
            raise FileNotFoundError(f"Missing PlantVillage color class directory: {color_cdir}")
        if not seg_cdir.exists():
            raise FileNotFoundError(f"Missing PlantVillage segmented class directory: {seg_cdir}")

        color_files = sorted(
            [f for f in color_cdir.iterdir() if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")],
            key=lambda p: p.name.lower(),
        )

        # Build segmented file lookup
        seg_lookup: Dict[str, Path] = {}
        for sf in seg_cdir.iterdir():
            if sf.is_file() and sf.suffix.lower() in (".jpg", ".jpeg", ".png"):
                seg_lookup[sf.stem.lower()] = sf
                seg_lookup[sf.stem.split("___")[0].lower()] = sf
                seg_suffix = sf.stem.split("___")[-1].replace("_final_masked", "").lower()
                seg_lookup[seg_suffix] = sf

        if len(color_files) < (n_train_200 + n_val):
            raise ValueError(
                f"Class {pv_class} only has {len(color_files)} images, need at least {n_train_200 + n_val}"
            )

        # Deterministic pre-cap for fast diversity feature extraction (retains >= 600 candidates)
        total_avail = len(color_files)
        if total_avail > 600:
            rng_pre = np.random.default_rng(seed + 500 + cidx)
            pre_idx = sorted(rng_pre.choice(total_avail, size=600, replace=False).tolist())
            candidate_files = [color_files[i] for i in pre_idx]
        else:
            candidate_files = color_files

        # Extract features for diversity selection
        feats = []
        matched_seg_paths = []
        for cf in candidate_files:
            feat, m, s = extract_lightweight_descriptor(cf)
            feats.append(feat)
            pv_means.append(m)
            pv_stds.append(s)
            matched_seg_paths.append(find_segmented_match(seg_cdir, cf, seg_lookup))

        feat_mat = np.stack(feats, axis=0)

        # Select 200 diverse medoids for PV_200
        idx_200 = select_diverse_medoids(feat_mat, k=n_train_200, seed=seed + cidx)
        set_200 = set(idx_200)

        # Select 100 diverse medoids strictly inside PV_200 for PV_100
        sub_feats_200 = feat_mat[idx_200]
        rel_idx_100 = select_diverse_medoids(sub_feats_200, k=n_train_100, seed=seed + 1000 + cidx)
        idx_100 = [idx_200[r] for r in rel_idx_100]
        set_100 = set(idx_100)

        # Select 50 validation samples strictly disjoint from PV_200
        remaining_indices = [i for i in range(len(candidate_files)) if i not in set_200]
        rng = np.random.default_rng(seed + 2000 + cidx)
        rng.shuffle(remaining_indices)
        idx_val = remaining_indices[:n_val]
        set_val = set(idx_val)

        # Sanity guarantees
        assert len(set_200) == n_train_200
        assert len(set_100) == n_train_100
        assert set_100.issubset(set_200)
        assert len(set_val) == n_val
        assert len(set_200 & set_val) == 0

        seg_matched_count = 0
        for idx in idx_200:
            seg_p = matched_seg_paths[idx]
            if seg_p is not None:
                seg_matched_count += 1
            rec = {
                "class_idx": cidx,
                "pd_class": pd_class,
                "pv_class": pv_class,
                "color_path": str(candidate_files[idx]),
                "segmented_path": seg_p,
            }
            pv_200_records.append(rec)
            if idx in set_100:
                pv_100_records.append(rec)

        for idx in idx_val:
            pv_val_records.append({
                "class_idx": cidx,
                "pd_class": pd_class,
                "pv_class": pv_class,
                "color_path": str(candidate_files[idx]),
                "segmented_path": matched_seg_paths[idx],
            })

        class_summary[pd_class] = {
            "class_idx": cidx,
            "pv_class": pv_class,
            "total_available": total_avail,
            "pv_200_count": len(idx_200),
            "pv_100_count": len(idx_100),
            "pv_val_count": len(idx_val),
            "segmented_match_rate": round(seg_matched_count / len(idx_200), 4),
        }
        logger.info(
            f"  [{cidx+1:02d}/27] {pd_class:28s} <- {pv_class:45s} | "
            f"Avail={total_avail:4d} | PV200={len(idx_200)} | PV100={len(idx_100)} | Val={len(idx_val)} | SegMatch={seg_matched_count}/{len(idx_200)}"
        )

    # Compute PlantDoc clean test RGB statistics for domain-shift quantification
    pd_means = []
    pd_stds = []
    if PD_CLEAN_TEST_DIR.exists():
        for pd_class in sorted_pd_classes:
            cdir = PD_CLEAN_TEST_DIR / pd_class
            if cdir.exists():
                for f in cdir.iterdir():
                    if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        _, m, s = extract_lightweight_descriptor(f)
                        pd_means.append(m)
                        pd_stds.append(s)

    pv_rgb_mean = np.mean(pv_means, axis=0).tolist()
    pv_rgb_std = np.mean(pv_stds, axis=0).tolist()
    pd_rgb_mean = np.mean(pd_means, axis=0).tolist() if pd_means else [0.0, 0.0, 0.0]
    pd_rgb_std = np.mean(pd_stds, axis=0).tolist() if pd_stds else [0.0, 0.0, 0.0]

    manifest = {
        "metadata": {
            "generator": "prepare_pv_subsets.py",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "random_seed": seed,
            "num_classes": len(sorted_pd_classes),
            "pv_200_total": len(pv_200_records),
            "pv_100_total": len(pv_100_records),
            "pv_val_total": len(pv_val_records),
            "zero_overlap_verified": True,
        },
        "domain_shift_rgb_stats": {
            "plantvillage_rgb_mean": [round(x, 4) for x in pv_rgb_mean],
            "plantvillage_rgb_std": [round(x, 4) for x in pv_rgb_std],
            "plantdoc_test_rgb_mean": [round(x, 4) for x in pd_rgb_mean],
            "plantdoc_test_rgb_std": [round(x, 4) for x in pd_rgb_std],
        },
        "classes": sorted_pd_classes,
        "pd_to_pv_class_map": PD_TO_PV_CLASS_MAP,
        "class_summary": class_summary,
        "splits": {
            "pv_200": pv_200_records,
            "pv_100": pv_100_records,
            "pv_val": pv_val_records,
        },
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = MANIFEST_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, MANIFEST_PATH)
    logger.info(f"Saved PlantVillage subset manifest atomically to {MANIFEST_PATH}")
    return manifest


if __name__ == "__main__":
    build_pv_subsets()
