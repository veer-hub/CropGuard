"""
Dataset Sanitization & Zero-Leakage GroupKFold Benchmark
Target: PlantDoc Augmented Dataset
Author: CropGuard v3 System Architecture
"""

import os
import sys
import re
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit

# Source and target directories
SOURCE_DIR = Path(r"c:\CropGuard v3\augmented_plantdoc")
TARGET_DIR = Path(r"c:\CropGuard v3\augmented_plantdoc_clean")
BASE_ID_PATTERN = re.compile(r'^(.+?\(\d+)\)?', re.IGNORECASE)
RANDOM_SEED = 47


def extract_base_id(filename: str) -> str:
    """Extract canonical base physical leaf ID by capturing up to the primary (<id>) stem.
    
    PlantDoc images follow the naming format '<class_prefix> (<id>)<suffix>.<ext>'.
    Suffixes and variations include:
      - Offline augmentation flags: '_aug_p0_18', '_aug', 'AUG_1', 'Aug_1'
      - Trailing crop / duplicate indices: '(18)1', '(18)2', '(3)1', '(3)2'
      - Unclosed parenthesis typos in upstream PlantDoc: '(3.jpg', '(67.jpg', '(8.jpg'
      - Multiple parentheses from nested augmentations: '(17)_aug_p2_62(2)'
    
    Extracting '{prefix} ({id})' guarantees that all variations and crops of the
    same physical leaf share the exact same base identifier.
    """
    stem = Path(filename).stem
    m = BASE_ID_PATTERN.match(stem)
    if m:
        return f"{m.group(1).strip()})"
    return stem.strip()


def sanitize_and_split():
    print("=" * 70)
    print("CGI-PlantDoc Dataset Sanitization & Zero-Leakage Split Engine")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Source Directory: {SOURCE_DIR}")
    print(f"Target Directory: {TARGET_DIR}")
    print("=" * 70)

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source directory does not exist: {SOURCE_DIR}")

    # Step 1: Scan and deduplicate raw files
    print("\n[Step 1] Scanning raw augmented_plantdoc repository...")
    raw_records = []
    seen_unique = set()
    total_scanned = 0

    for split in ["train", "val", "test"]:
        sdir = SOURCE_DIR / split
        if not sdir.exists():
            continue
        for cdir in sdir.iterdir():
            if not cdir.is_dir():
                continue
            cname = cdir.name
            for f in cdir.iterdir():
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    total_scanned += 1
                    target_cname = cname
                    # Remediate misfiled late blight image
                    if f.name == "Potato leaf late blight (73)_aug_p0_1.jpg" and cname == "Potato leaf early blight":
                        target_cname = "Potato leaf late blight"

                    key = (target_cname, f.name)
                    if key in seen_unique:
                        continue
                    seen_unique.add(key)

                    base_id = extract_base_id(f.name)
                    raw_records.append({
                        "class": target_cname,
                        "filename": f.name,
                        "base_id": base_id,
                        "source_path": f,
                        "original_split": split,
                        "original_class": cname
                    })

    duplicates_removed = total_scanned - len(raw_records)
    print(f"  -> Total files scanned: {total_scanned}")
    print(f"  -> Redundant cross-split duplicates removed: {duplicates_removed}")
    print(f"  -> Unique physical / augmented images retained: {len(raw_records)}")

    # Step 2: Organize by class and base image ID
    print("\n[Step 2] Grouping by canonical base image IDs...")
    class_groups = defaultdict(lambda: defaultdict(list))
    for r in raw_records:
        class_groups[r["class"]][r["base_id"]].append(r)

    all_classes = sorted(list(class_groups.keys()))
    print(f"  -> Total unique classes detected: {len(all_classes)}")

    total_bases = sum(len(bases) for bases in class_groups.values())
    print(f"  -> Total unique canonical base leaf IDs: {total_bases}")

    # Step 3: Perform GroupShuffleSplit per class (70% train, 15% val, 15% test)
    print("\n[Step 3] Executing strict GroupShuffleSplit partitioning (70/15/15)...")
    split_assignments = {"train": [], "val": [], "test": []}
    class_split_stats = {}

    for cname in all_classes:
        c_records = [r for r in raw_records if r["class"] == cname]
        groups = [r["base_id"] for r in c_records]
        unique_bases = sorted(list(set(groups)))
        n_bases = len(unique_bases)

        # Split 1: 70% Train, 30% Remainder (Val + Test)
        gss_train = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=RANDOM_SEED)
        train_idx, rem_idx = next(gss_train.split(c_records, groups=groups))

        rem_records = [c_records[i] for i in rem_idx]
        rem_groups = [groups[i] for i in rem_idx]

        # Split 2: 50% Val, 50% Test of Remainder (= 15% and 15% of total)
        gss_val = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=RANDOM_SEED)
        val_sub_idx, test_sub_idx = next(gss_val.split(rem_records, groups=rem_groups))

        c_train = [c_records[i] for i in train_idx]
        c_val = [rem_records[i] for i in val_sub_idx]
        c_test = [rem_records[i] for i in test_sub_idx]

        train_bases = set(r["base_id"] for r in c_train)
        val_bases = set(r["base_id"] for r in c_val)
        test_bases = set(r["base_id"] for r in c_test)

        # Integrity checks
        assert len(train_bases & val_bases) == 0, f"Leakage detected between train and val for {cname}"
        assert len(train_bases & test_bases) == 0, f"Leakage detected between train and test for {cname}"
        assert len(val_bases & test_bases) == 0, f"Leakage detected between val and test for {cname}"
        assert len(c_train) > 0, f"Train split empty for {cname}"
        assert len(c_val) > 0, f"Val split empty for {cname}"
        assert len(c_test) > 0, f"Test split empty for {cname}"

        for r in c_train:
            r["target_split"] = "train"
        for r in c_val:
            r["target_split"] = "val"
        for r in c_test:
            r["target_split"] = "test"

        split_assignments["train"].extend(c_train)
        split_assignments["val"].extend(c_val)
        split_assignments["test"].extend(c_test)

        class_split_stats[cname] = {
            "total_images": len(c_records),
            "total_base_ids": n_bases,
            "train": {"images": len(c_train), "base_ids": len(train_bases)},
            "val": {"images": len(c_val), "base_ids": len(val_bases)},
            "test": {"images": len(c_test), "base_ids": len(test_bases)},
            "leakage": {
                "train_val_base_overlap": len(train_bases & val_bases),
                "train_test_base_overlap": len(train_bases & test_bases),
                "val_test_base_overlap": len(val_bases & test_bases)
            }
        }

        print(f"  {cname:30s} | Bases: Tr={len(train_bases):3d}, Val={len(val_bases):2d}, Te={len(test_bases):2d} "
              f"| Imgs: Tr={len(c_train):3d}, Val={len(c_val):3d}, Te={len(c_test):3d} (Total={len(c_records)})")

    # Step 4: Write sanitized dataset to target directory
    print(f"\n[Step 4] Materializing clean dataset to {TARGET_DIR}...")
    if TARGET_DIR.exists():
        print(f"  -> Cleaning pre-existing directory: {TARGET_DIR}")
        shutil.rmtree(TARGET_DIR)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    hard_links_created = 0
    copies_created = 0

    for split in ["train", "val", "test"]:
        records_to_copy = split_assignments[split]
        for r in records_to_copy:
            dest_dir = TARGET_DIR / split / r["class"]
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / r["filename"]

            src_path = r["source_path"]
            try:
                os.link(src_path, dest_path)
                hard_links_created += 1
            except Exception:
                shutil.copy2(src_path, dest_path)
                copies_created += 1

    print(f"  -> Materialization complete: {hard_links_created} hard links, {copies_created} copies.")

    # Step 5: Post-creation verification directly from filesystem
    print("\n[Step 5] Running strict disk-level verification check...")
    disk_bases = defaultdict(lambda: defaultdict(set))
    disk_counts = defaultdict(lambda: defaultdict(int))

    for split in ["train", "val", "test"]:
        sdir = TARGET_DIR / split
        if not sdir.exists():
            raise RuntimeError(f"Split directory {sdir} missing on disk!")
        for cdir in sdir.iterdir():
            if not cdir.is_dir():
                continue
            cname = cdir.name
            for f in cdir.iterdir():
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    b_id = extract_base_id(f.name)
                    disk_bases[split][cname].add(b_id)
                    disk_counts[split][cname] += 1

    # Verify all 27 classes exist in all 3 splits on disk
    for split in ["train", "val", "test"]:
        missing_classes = set(all_classes) - set(disk_counts[split].keys())
        if missing_classes:
            raise RuntimeError(f"Split {split} is missing classes: {missing_classes}")
        print(f"  -> Split '{split}': All {len(disk_counts[split])} classes present on disk.")

    # Global base ID sets
    disk_train_bases = set()
    disk_val_bases = set()
    disk_test_bases = set()

    for cname in all_classes:
        disk_train_bases.update(disk_bases["train"][cname])
        disk_val_bases.update(disk_bases["val"][cname])
        disk_test_bases.update(disk_bases["test"][cname])

    leakage_train_val = disk_train_bases & disk_val_bases
    leakage_train_test = disk_train_bases & disk_test_bases
    leakage_val_test = disk_val_bases & disk_test_bases

    print(f"  -> Disk Check: BaseIDs(Train) & BaseIDs(Val)  = {len(leakage_train_val)}")
    print(f"  -> Disk Check: BaseIDs(Train) & BaseIDs(Test) = {len(leakage_train_test)}")
    print(f"  -> Disk Check: BaseIDs(Val)   & BaseIDs(Test) = {len(leakage_val_test)}")

    if len(leakage_train_val) > 0 or len(leakage_train_test) > 0 or len(leakage_val_test) > 0:
        raise RuntimeError("FATAL: Zero-leakage invariant violated on disk!")
    print("  -> ZERO-LEAKAGE GUARANTEE CONFIRMED: All partitions are strictly disjoint at base-image level!")

    # Step 6: Generate and write sanitization report
    print("\n[Step 6] Compiling sanitization_report.json...")
    total_clean_images = sum(len(split_assignments[s]) for s in ["train", "val", "test"])

    report = {
        "metadata": {
            "generator": "sanitize_plantdoc_splits.py",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_directory": str(SOURCE_DIR),
            "target_directory": str(TARGET_DIR),
            "random_seed": RANDOM_SEED,
            "target_split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15}
        },
        "dataset_summary": {
            "total_classes": len(all_classes),
            "raw_scanned_files": total_scanned,
            "redundant_cross_split_duplicates_removed": duplicates_removed,
            "unique_clean_images": total_clean_images,
            "unique_canonical_base_leaf_ids": total_bases,
            "materialization_method": {
                "hard_links": hard_links_created,
                "file_copies": copies_created
            }
        },
        "partition_totals": {
            "train": {
                "image_count": len(split_assignments["train"]),
                "image_percentage": round(len(split_assignments["train"]) / total_clean_images * 100, 2),
                "base_id_count": len(disk_train_bases),
                "base_id_percentage": round(len(disk_train_bases) / total_bases * 100, 2)
            },
            "val": {
                "image_count": len(split_assignments["val"]),
                "image_percentage": round(len(split_assignments["val"]) / total_clean_images * 100, 2),
                "base_id_count": len(disk_val_bases),
                "base_id_percentage": round(len(disk_val_bases) / total_bases * 100, 2)
            },
            "test": {
                "image_count": len(split_assignments["test"]),
                "image_percentage": round(len(split_assignments["test"]) / total_clean_images * 100, 2),
                "base_id_count": len(disk_test_bases),
                "base_id_percentage": round(len(disk_test_bases) / total_bases * 100, 2)
            }
        },
        "zero_leakage_audit": {
            "train_val_base_overlap": len(leakage_train_val),
            "train_test_base_overlap": len(leakage_train_test),
            "val_test_base_overlap": len(leakage_val_test),
            "zero_leakage_verified": True,
            "all_classes_present_in_train": len(disk_counts["train"]) == 27,
            "all_classes_present_in_val": len(disk_counts["val"]) == 27,
            "all_classes_present_in_test": len(disk_counts["test"]) == 27
        },
        "formerly_missing_classes_resolution": {
            "Tomato leaf mosaic virus": {
                "prior_state": "280 images in train, 0 in val, 0 in test",
                "sanitized_state": class_split_stats["Tomato leaf mosaic virus"]
            },
            "Tomato leaf yellow virus": {
                "prior_state": "280 images in train, 0 in val, 0 in test",
                "sanitized_state": class_split_stats["Tomato leaf yellow virus"]
            }
        },
        "per_class_breakdown": class_split_stats
    }

    report_path = TARGET_DIR / "sanitization_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"  -> Report written to {report_path}")
    print("\nDataset sanitization and validation completed successfully!")
    return report


if __name__ == "__main__":
    sanitize_and_split()
