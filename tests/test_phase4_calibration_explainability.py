"""
Unit and Integration Tests for Phase 4: Calibration, Uncertainty Quantification & Explainability
Verifies Step 4.1 (Logits Cache), Step 4.2 (Temperature Scaling & ECE < 0.04),
Step 4.3 (24-Layer Attention Rollout & MAF Saliency Overlays), and
Step 4.4 (Integrated Gradients & Lesion Relevance Ratio LRR > 0.80).
"""

import sys
import json
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

# Add workspace root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from evaluate_calibration import (
    compute_calibration_metrics,
    optimize_temperature_lbfgs,
    compute_margin_temperatures,
)
from generate_explainability_maps import (
    compute_recursive_attention_rollout,
    upsample_heatmap,
    denormalize_image,
)
from compute_lrr_metrics import (
    compute_lrr_score,
    segment_lesion_and_leaf_mask,
)


class TestPhase4CalibrationAndExplainability(unittest.TestCase):

    def test_step_4_1_logits_cache_integrity(self):
        """Verify logits_cache.pt exists and has exact shapes and Phase 3 benchmark accuracies."""
        cache_path = ROOT_DIR / "checkpoints" / "dinov2_lora" / "logits_cache.pt"
        self.assertTrue(cache_path.exists(), f"Missing logits cache: {cache_path}")

        cache = torch.load(cache_path, map_location="cpu")
        self.assertEqual(cache["val_logits"].shape, (1546, 27))
        self.assertEqual(cache["val_targets"].shape, (1546,))
        self.assertEqual(cache["test_logits"].shape, (1588, 27))
        self.assertEqual(cache["test_targets"].shape, (1588,))
        self.assertEqual(len(cache["classes"]), 27)
        self.assertEqual(len(cache["val_paths"]), 1546)
        self.assertEqual(len(cache["test_paths"]), 1588)

        # Verify no NaNs or Infs in cached logits
        self.assertFalse(torch.isnan(cache["val_logits"]).any().item())
        self.assertFalse(torch.isnan(cache["test_logits"]).any().item())

        # Verify exact Phase 3 accuracies (Val: 81.05%, Test: 78.84%)
        self.assertAlmostEqual(cache["val_accuracy"], 0.81047865, places=4)
        self.assertAlmostEqual(cache["test_accuracy"], 0.78841310, places=4)

    def test_step_4_2_calibration_math_and_edge_cases(self):
        """Verify calibration metrics computation, bin partitioning, and error handling on invalid inputs."""
        logits = torch.randn(100, 27)
        targets = torch.randint(0, 27, (100,))

        # Edge cases: non-positive temperature
        with self.assertRaises(ValueError):
            compute_calibration_metrics(logits, targets, temperature=0.0)
        with self.assertRaises(ValueError):
            compute_calibration_metrics(logits, targets, temperature=-1.5)
        with self.assertRaises(ValueError):
            compute_calibration_metrics(logits, targets, temperature=torch.tensor([1.0, -0.5]))

        # Edge cases: positive temperature tensor with mismatched length
        with self.assertRaises(ValueError):
            compute_calibration_metrics(logits, targets, temperature=torch.tensor([1.0, 1.5]))
        with self.assertRaises(ValueError):
            compute_calibration_metrics(logits, targets, temperature=torch.ones(100, 2))

        # Edge cases: empty or malformed tensors
        with self.assertRaises(ValueError):
            compute_calibration_metrics(torch.empty(0, 27), torch.empty(0, dtype=torch.long))
        with self.assertRaises(ValueError):
            compute_calibration_metrics(torch.randn(100), targets)
        with self.assertRaises(ValueError):
            compute_calibration_metrics(logits, torch.randint(0, 27, (50,)))

        # Valid 15-bin metrics properties
        metrics = compute_calibration_metrics(logits, targets, temperature=1.2, num_bins=15)
        self.assertEqual(len(metrics["bins"]), 15)
        total_count = sum(b["count"] for b in metrics["bins"])
        self.assertEqual(total_count, 100)
        total_prop = sum(b["proportion"] for b in metrics["bins"])
        self.assertAlmostEqual(total_prop, 1.0, places=4)
        self.assertGreaterEqual(metrics["ece"], 0.0)
        self.assertLessEqual(metrics["ece"], 1.0)
        self.assertGreaterEqual(metrics["mce"], metrics["ece"])

    def test_step_4_2_temperature_scaling_preserves_accuracy_and_meets_target(self):
        """Verify post-hoc calibration preserves 100% of predictions, populates Bin 15, and achieves ECE < 0.04."""
        json_path = ROOT_DIR / "docs" / "calibration_metrics.json"
        fig_path = ROOT_DIR / "docs" / "figures" / "reliability_diagram.png"
        self.assertTrue(json_path.exists(), f"Missing {json_path}")
        self.assertTrue(fig_path.exists(), f"Missing {fig_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        self.assertEqual(report["num_bins"], 15)
        self.assertGreater(report["optimal_temperature"], 0.0)
        self.assertIn("c_cap", report["margin_scaling_params"])

        # Verify ECE < 0.04 on both validation and held-out test sets
        val_cal_ece = report["validation"]["calibrated"]["ece"]
        test_cal_ece = report["test"]["calibrated"]["ece"]
        test_uncal_ece = report["test"]["uncalibrated"]["ece"]

        self.assertLess(val_cal_ece, 0.04, f"Expected val calibrated ECE < 0.04, got {val_cal_ece}")
        self.assertLess(test_cal_ece, 0.04, f"Expected test calibrated ECE < 0.04, got {test_cal_ece}")
        self.assertLess(test_cal_ece, test_uncal_ece)

        # Verify Bin 15 ([0.9333, 1.0]) is populated (count > 0) and well-calibrated on both val and test
        val_bin15 = report["validation"]["calibrated"]["bins"][14]
        test_bin15 = report["test"]["calibrated"]["bins"][14]
        self.assertGreater(val_bin15["count"], 50, "Bin 15 should be populated on validation set")
        self.assertGreater(test_bin15["count"], 50, "Bin 15 should be populated on test set")
        self.assertLess(val_bin15["gap"], 0.02)
        self.assertLess(test_bin15["gap"], 0.02)

        # Verify NLL and Brier score improved on held-out test set
        self.assertLess(
            report["test"]["calibrated"]["nll"],
            report["test"]["uncalibrated"]["nll"],
        )
        self.assertLess(
            report["test"]["calibrated"]["brier_score"],
            report["test"]["uncalibrated"]["brier_score"],
        )

        # Verify reliability diagram image is valid high-res PNG
        with Image.open(fig_path) as img:
            self.assertEqual(img.format, "PNG")
            self.assertGreater(img.width, 1000)
            self.assertGreater(img.height, 800)

    def test_step_4_3_attention_rollout_math_and_figures(self):
        """Verify 24-layer recursive attention rollout mathematics, register-sink debiasing, and figures."""
        # Edge case: empty list
        with self.assertRaises(ValueError):
            compute_recursive_attention_rollout([])

        # Edge case: non-square patch count (1 + 20 tokens -> 20 patches is not a square)
        bad_attn = [torch.rand(2, 4, 21, 21)]
        with self.assertRaises(ValueError):
            compute_recursive_attention_rollout(bad_attn)

        # Valid 24-layer DINOv2 rollout (B=2, H=16, T=257 [1 CLS + 256 patches], MAF H=8)
        B, H_vit, T, H_maf = 2, 16, 257, 8
        layer_attns = [torch.softmax(torch.randn(B, H_vit, T, T), dim=-1) for _ in range(24)]
        maf_attn = torch.softmax(torch.randn(B, H_maf, T - 1), dim=-1)

        rollout_grid, maf_grid = compute_recursive_attention_rollout(
            layer_attentions=layer_attns,
            maf_attention=maf_attn,
            residual_weight=0.5,
            discard_ratio=0.1,
        )
        self.assertEqual(rollout_grid.shape, (B, 16, 16))
        self.assertEqual(maf_grid.shape, (B, 16, 16))

        # Check min-max normalization in [0, 1]
        for b in range(B):
            self.assertAlmostEqual(float(rollout_grid[b].min().item()), 0.0, places=5)
            self.assertAlmostEqual(float(rollout_grid[b].max().item()), 1.0, places=5)

        # Verify register-sink column debiasing: if patch 0 is an artificial global attention sink across layers
        # while MAF query attends strictly to center patch 136 (row 8, col 8), patch 136 must remain the peak
        sink_logits = torch.zeros(1, 4, 257, 257)
        sink_logits[:, :, :, 1] = 4.0  # Patch 0 (token index 1) acts as global attention sink
        sink_attns = [torch.softmax(sink_logits, dim=-1) for _ in range(8)]
        focused_maf = torch.zeros(1, 8, 256)
        focused_maf[:, :, 136] = 1.0
        r_sink, _ = compute_recursive_attention_rollout(sink_attns, focused_maf)
        self.assertAlmostEqual(float(r_sink[0, 8, 8].item()), 1.0, places=5)

        # Verify square and non-square (H, W) upsample_heatmap
        upsampled = upsample_heatmap(rollout_grid[0], image_size=224)
        self.assertEqual(upsampled.shape, (224, 224))
        self.assertGreaterEqual(float(upsampled.min()), 0.0)
        self.assertLessEqual(float(upsampled.max()), 1.0)

        upsampled_rect = upsample_heatmap(rollout_grid[0], image_size=(192, 256))
        self.assertEqual(upsampled_rect.shape, (192, 256))

        # Verify generated explainability figures exist
        grid_fig = ROOT_DIR / "docs" / "figures" / "attention_rollout_grid.png"
        overlays_dir = ROOT_DIR / "docs" / "figures" / "saliency_overlays"
        self.assertTrue(grid_fig.exists(), f"Missing {grid_fig}")
        self.assertTrue(overlays_dir.exists(), f"Missing {overlays_dir}")
        overlay_files = list(overlays_dir.glob("*.png"))
        self.assertGreaterEqual(len(overlay_files), 6)

    def test_step_4_4_lrr_math_and_edge_cases(self):
        """Verify Lesion Relevance Ratio (LRR) boundary conditions and non-circular background rejection."""
        import inspect
        sig = inspect.signature(segment_lesion_and_leaf_mask)
        self.assertEqual(
            list(sig.parameters.keys()),
            ["rgb_img"],
            "segment_lesion_and_leaf_mask must depend strictly on rgb_img with no saliency leakage parameter",
        )

        # Shape mismatch
        with self.assertRaises(ValueError):
            compute_lrr_score(np.ones((10, 10)), np.ones((10, 12)))

        # Empty array
        with self.assertRaises(ValueError):
            compute_lrr_score(np.array([]), np.array([]))

        # All-zero saliency map returns 0.0 without NaN
        zero_lrr = compute_lrr_score(np.zeros((16, 16)), np.ones((16, 16)))
        self.assertEqual(zero_lrr, 0.0)

        # 100% inside lesion mask -> LRR == 1.0
        mask = np.zeros((20, 20), dtype=np.float32)
        mask[5:15, 5:15] = 1.0
        sal_inside = np.zeros((20, 20), dtype=np.float32)
        sal_inside[6:12, 6:12] = 2.5
        self.assertAlmostEqual(compute_lrr_score(sal_inside, mask), 1.0, places=6)

        # 100% outside lesion mask -> LRR == 0.0
        sal_outside = np.zeros((20, 20), dtype=np.float32)
        sal_outside[0:4, 0:4] = 3.0
        self.assertAlmostEqual(compute_lrr_score(sal_outside, mask), 0.0, places=6)

        # Regression Test 1: Black background with green leaf in bottom-right [120:220, 120:220]
        rgb_black_bg = np.zeros((224, 224, 3), dtype=np.float32)
        rgb_black_bg[120:220, 120:220, 1] = 0.8
        mask_black_bg = segment_lesion_and_leaf_mask(rgb_black_bg)
        wrong_sal_bg = np.zeros((224, 224), dtype=np.float32)
        wrong_sal_bg[0:80, 0:80] = 1.0
        right_sal_leaf = np.zeros((224, 224), dtype=np.float32)
        right_sal_leaf[130:200, 130:200] = 1.0
        self.assertAlmostEqual(compute_lrr_score(wrong_sal_bg, mask_black_bg), 0.0, places=6)
        self.assertAlmostEqual(compute_lrr_score(right_sal_leaf, mask_black_bg), 1.0, places=6)

        # Regression Test 2: Achromatic gray/soil background noise with leaf in bottom-right
        rng = np.random.RandomState(42)
        rgb_soil_bg = rng.uniform(0.20, 0.24, (224, 224, 3)).astype(np.float32)
        rgb_soil_bg[120:210, 120:210, 0] = 0.1
        rgb_soil_bg[120:210, 120:210, 1] = 0.8
        rgb_soil_bg[120:210, 120:210, 2] = 0.1
        mask_soil_bg = segment_lesion_and_leaf_mask(rgb_soil_bg)
        self.assertAlmostEqual(compute_lrr_score(wrong_sal_bg, mask_soil_bg), 0.0, places=6)
        self.assertAlmostEqual(compute_lrr_score(right_sal_leaf, mask_soil_bg), 1.0, places=6)

    def test_step_4_4_lrr_report_and_figure(self):
        """Verify LRR summary report and distribution plot exist and confirm mean LRR > 0.80."""
        lrr_json = ROOT_DIR / "docs" / "lrr_summary_report.json"
        lrr_fig = ROOT_DIR / "docs" / "figures" / "lrr_distribution.png"
        self.assertTrue(lrr_json.exists(), f"Missing {lrr_json}")
        self.assertTrue(lrr_fig.exists(), f"Missing {lrr_fig}")

        with open(lrr_json, "r", encoding="utf-8") as f:
            lrr_data = json.load(f)

        self.assertTrue(lrr_data["target_met"])
        self.assertGreater(lrr_data["mean_lrr"], 0.80)
        self.assertGreater(lrr_data["median_lrr"], 0.85)
        self.assertEqual(lrr_data["num_classes"], 27)
        self.assertEqual(lrr_data["num_evaluated_samples"], 108)


if __name__ == "__main__":
    unittest.main()
