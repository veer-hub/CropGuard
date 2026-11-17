"""
Unit and Integration Tests for Phase 5: Cross-Domain Lab-to-Field Transfer
CropGuard v3 - PlantVillage (PV_200 & PV_100) -> PlantDoc Zero-Shot Transfer
"""

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from train_cross_domain_pv import (
    UnlabeledTargetFieldDataset,
    composite_segmented_leaf,
    deep_coral_loss,
    generate_synthetic_field_bg,
    get_target_weak_strong_transforms,
    shot_information_maximization_loss,
)

MANIFEST_PATH = BASE_DIR / "docs" / "pv_subsets_manifest.json"
RESULTS_PATH = BASE_DIR / "docs" / "cross_domain_results.json"
FIG_PATH = BASE_DIR / "docs" / "figures" / "cross_domain_comparison.png"
PD_CLEAN_DIR = BASE_DIR / "augmented_plantdoc_clean"


class TestPhase5CrossDomain(unittest.TestCase):

    def test_pv_subsets_manifest_integrity_and_disjointness(self):
        """Verify PV_200, PV_100, and PV_Val counts, class parity, and zero overlap."""
        self.assertTrue(MANIFEST_PATH.exists(), f"Missing manifest: {MANIFEST_PATH}")
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        classes = manifest["classes"]
        self.assertEqual(len(classes), 27)
        pd_test_classes = sorted([d.name for d in (PD_CLEAN_DIR / "test").iterdir() if d.is_dir()])
        self.assertEqual(classes, pd_test_classes)

        pv_200 = manifest["splits"]["pv_200"]
        pv_100 = manifest["splits"]["pv_100"]
        pv_val = manifest["splits"]["pv_val"]

        self.assertEqual(len(pv_200), 200 * 27)
        self.assertEqual(len(pv_100), 100 * 27)
        self.assertEqual(len(pv_val), 50 * 27)

        paths_200 = {r["color_path"] for r in pv_200}
        paths_100 = {r["color_path"] for r in pv_100}
        paths_val = {r["color_path"] for r in pv_val}

        self.assertTrue(paths_100.issubset(paths_200), "PV_100 must be a strict subset of PV_200")
        self.assertEqual(len(paths_200 & paths_val), 0, "PV_200 and PV_Val must be strictly disjoint")

        for cls_name, info in manifest["class_summary"].items():
            self.assertEqual(info["pv_200_count"], 200)
            self.assertEqual(info["pv_100_count"], 100)
            self.assertEqual(info["pv_val_count"], 50)
            self.assertEqual(info["segmented_match_rate"], 1.0, f"Segmented match rate < 1.0 for {cls_name}")

    def test_synthetic_background_and_segmented_compositing(self):
        """Verify synthetic field background generation and leaf compositing."""
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        sample = manifest["splits"]["pv_200"][0]
        rng = np.random.default_rng(47)
        bg = generate_synthetic_field_bg(256, rng)
        self.assertEqual(bg.shape, (256, 256, 3))
        self.assertEqual(bg.dtype, np.uint8)
        self.assertGreater(bg.std(), 2.0, "Background should have spatial variation")

        comp = composite_segmented_leaf(sample["color_path"], sample["segmented_path"], rng, canvas_size=256)
        self.assertEqual(comp.shape, (256, 256, 3))
        self.assertEqual(comp.dtype, np.uint8)

    def test_coral_and_shot_losses(self):
        """Verify Deep CORAL covariance alignment and SHOT IM losses."""
        s = torch.randn(16, 64, requires_grad=True)
        t_same = s.detach().clone()
        self.assertAlmostEqual(float(deep_coral_loss(s, t_same).item()), 0.0, places=5)

        t_diff = torch.randn(16, 64) * 3.0 + 2.0
        loss_c = deep_coral_loss(s, t_diff)
        self.assertGreater(float(loss_c.item()), 0.1)
        loss_c.backward()
        self.assertIsNotNone(s.grad)

        logits = torch.randn(16, 27, requires_grad=True)
        loss_im = shot_information_maximization_loss(logits)
        self.assertTrue(torch.isfinite(loss_im))

    def test_unlabeled_target_dataset_excludes_val_and_test(self):
        """Verify UnlabeledTargetFieldDataset rejects val/test directories to enforce zero leakage."""
        weak_tf, strong_tf = get_target_weak_strong_transforms()
        with self.assertRaises(ValueError):
            UnlabeledTargetFieldDataset(PD_CLEAN_DIR / "test", weak_tf, strong_tf)
        with self.assertRaises(ValueError):
            UnlabeledTargetFieldDataset(PD_CLEAN_DIR / "val", weak_tf, strong_tf)

        ds = UnlabeledTargetFieldDataset(PD_CLEAN_DIR / "train", weak_tf, strong_tf)
        self.assertEqual(len(ds), 7158)
        for p in ds.image_paths:
            self.assertEqual(p.parent.parent.name, "train")

    def test_cross_domain_results_beat_salman_benchmarks(self):
        """Verify PV_200 and PV_100 beat Salman et al. 2025 (68%/0.65 and 49%/0.51) and Stage A < B < C."""
        self.assertTrue(RESULTS_PATH.exists(), f"Missing results file: {RESULTS_PATH}")

        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            res = json.load(f)

        self.assertEqual(res["target_num_samples"], 1588)
        self.assertEqual(res["num_classes"], 27)

        pv200_stages = res["subsets"]["pv_200"]["stages"]
        pv100_stages = res["subsets"]["pv_100"]["stages"]

        # Verify ablation progression: Stage A < Stage B < Stage C
        self.assertLess(pv200_stages["a"]["plantdoc_test_accuracy"], pv200_stages["b"]["plantdoc_test_accuracy"])
        self.assertLess(pv200_stages["b"]["plantdoc_test_accuracy"], pv200_stages["c"]["plantdoc_test_accuracy"])
        self.assertLess(pv100_stages["a"]["plantdoc_test_accuracy"], pv100_stages["b"]["plantdoc_test_accuracy"])
        self.assertLess(pv100_stages["b"]["plantdoc_test_accuracy"], pv100_stages["c"]["plantdoc_test_accuracy"])

        # Beat Salman et al. Table 4 (68.00% Acc, 0.65 F1) and Table 5 (49.00% Acc, 0.51 F1)
        self.assertGreater(pv200_stages["c"]["plantdoc_test_accuracy"], 0.68, "PV_200 -> PlantDoc must beat Salman 68%")
        self.assertGreater(pv200_stages["c"]["plantdoc_test_macro_f1"], 0.65, "PV_200 -> PlantDoc F1 must beat Salman 0.65")
        self.assertGreater(pv100_stages["c"]["plantdoc_test_accuracy"], 0.49, "PV_100 -> PlantDoc must beat Salman 49%")
        self.assertGreater(pv100_stages["c"]["plantdoc_test_macro_f1"], 0.51, "PV_100 -> PlantDoc F1 must beat Salman 0.51")
        self.assertTrue(FIG_PATH.exists() and FIG_PATH.stat().st_size > 10000)


if __name__ == "__main__":
    unittest.main()
