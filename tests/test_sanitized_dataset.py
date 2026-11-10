"""
Unit and Integration Tests for CGI-PlantDoc Dataset Sanitization
Tests zero base-image leakage, partition distribution, and data integrity.
"""

import unittest
import json
import re
from pathlib import Path
from PIL import Image

CLEAN_DIR = Path(r"c:\CropGuard v3\augmented_plantdoc_clean")
REPORT_PATH = CLEAN_DIR / "sanitization_report.json"
BASE_ID_PATTERN = re.compile(r'^(.+?\(\d+)\)?', re.IGNORECASE)


def extract_base_id(filename: str) -> str:
    stem = Path(filename).stem
    m = BASE_ID_PATTERN.match(stem)
    if m:
        return f"{m.group(1).strip()})"
    return stem.strip()


class TestSanitizedDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.splits = ["train", "val", "test"]
        cls.clean_dir = CLEAN_DIR
        cls.report_path = REPORT_PATH
        
        # Collect all files directly from disk
        cls.files_by_split = {s: [] for s in cls.splits}
        cls.bases_by_split = {s: set() for s in cls.splits}
        cls.classes_by_split = {s: set() for s in cls.splits}
        cls.class_files = {s: {} for s in cls.splits}
        cls.class_bases = {s: {} for s in cls.splits}
        
        for s in cls.splits:
            sdir = cls.clean_dir / s
            cls.assertTrue(sdir.exists(), f"Split directory missing: {sdir}")
            for cdir in sdir.iterdir():
                if not cdir.is_dir():
                    continue
                cname = cdir.name
                cls.classes_by_split[s].add(cname)
                cls.class_files[s][cname] = []
                cls.class_bases[s][cname] = set()
                
                for f in cdir.iterdir():
                    if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        cls.files_by_split[s].append((cname, f.name, f))
                        cls.class_files[s][cname].append(f)
                        b_id = extract_base_id(f.name)
                        cls.bases_by_split[s].add(b_id)
                        cls.class_bases[s][cname].add(b_id)

    def test_directory_structure(self):
        """Verify train, val, and test root directories exist."""
        for s in self.splits:
            self.assertTrue((self.clean_dir / s).is_dir(), f"Missing directory: {self.clean_dir / s}")

    def test_all_27_classes_present_in_all_splits(self):
        """Verify that exactly 27 classes exist in train, val, and test."""
        for s in self.splits:
            classes = self.classes_by_split[s]
            self.assertEqual(len(classes), 27, f"Split {s} has {len(classes)} classes, expected 27.")
            
        # Ensure class names match across splits
        self.assertEqual(self.classes_by_split["train"], self.classes_by_split["val"])
        self.assertEqual(self.classes_by_split["train"], self.classes_by_split["test"])

    def test_zero_leakage_global(self):
        """Verify that BaseIDs(Train) ∩ BaseIDs(Val) == ∅ and BaseIDs(Train) ∩ BaseIDs(Test) == ∅."""
        train_bases = self.bases_by_split["train"]
        val_bases = self.bases_by_split["val"]
        test_bases = self.bases_by_split["test"]

        train_val_overlap = train_bases.intersection(val_bases)
        train_test_overlap = train_bases.intersection(test_bases)
        val_test_overlap = val_bases.intersection(test_bases)

        self.assertEqual(len(train_val_overlap), 0, f"Leakage Train ∩ Val: {train_val_overlap}")
        self.assertEqual(len(train_test_overlap), 0, f"Leakage Train ∩ Test: {train_test_overlap}")
        self.assertEqual(len(val_test_overlap), 0, f"Leakage Val ∩ Test: {val_test_overlap}")

    def test_zero_leakage_per_class(self):
        """Verify zero base-image leakage within every single class individually."""
        for cname in self.classes_by_split["train"]:
            c_train_bases = self.class_bases["train"][cname]
            c_val_bases = self.class_bases["val"][cname]
            c_test_bases = self.class_bases["test"][cname]

            self.assertEqual(len(c_train_bases & c_val_bases), 0,
                             f"Class {cname} has base overlap between train and val")
            self.assertEqual(len(c_train_bases & c_test_bases), 0,
                             f"Class {cname} has base overlap between train and test")
            self.assertEqual(len(c_val_bases & c_test_bases), 0,
                             f"Class {cname} has base overlap between val and test")

            # Verify non-empty partitions
            self.assertGreater(len(c_train_bases), 0, f"Class {cname} has 0 train bases")
            self.assertGreater(len(c_val_bases), 0, f"Class {cname} has 0 val bases")
            self.assertGreater(len(c_test_bases), 0, f"Class {cname} has 0 test bases")

    def test_tomato_viruses_properly_partitioned(self):
        """Verify Tomato leaf mosaic virus and Tomato leaf yellow virus exist across all 3 splits."""
        for virus_class in ["Tomato leaf mosaic virus", "Tomato leaf yellow virus"]:
            tr_imgs = len(self.class_files["train"][virus_class])
            val_imgs = len(self.class_files["val"][virus_class])
            te_imgs = len(self.class_files["test"][virus_class])
            tot = tr_imgs + val_imgs + te_imgs

            self.assertEqual(tot, 280, f"{virus_class} total images should be 280, got {tot}")
            self.assertGreater(tr_imgs, 0, f"{virus_class} missing train images")
            self.assertGreater(val_imgs, 0, f"{virus_class} missing val images")
            self.assertGreater(te_imgs, 0, f"{virus_class} missing test images")

            # Base counts
            tr_b = len(self.class_bases["train"][virus_class])
            val_b = len(self.class_bases["val"][virus_class])
            te_b = len(self.class_bases["test"][virus_class])
            self.assertGreater(tr_b, 0)
            self.assertGreater(val_b, 0)
            self.assertGreater(te_b, 0)
            self.assertEqual(len(self.class_bases["train"][virus_class] & self.class_bases["val"][virus_class]), 0)
            self.assertEqual(len(self.class_bases["train"][virus_class] & self.class_bases["test"][virus_class]), 0)
            self.assertEqual(len(self.class_bases["val"][virus_class] & self.class_bases["test"][virus_class]), 0)

    def test_no_duplicate_files_across_splits(self):
        """Verify no filename appears in multiple splits."""
        train_files = set(f[1] for f in self.files_by_split["train"])
        val_files = set(f[1] for f in self.files_by_split["val"])
        test_files = set(f[1] for f in self.files_by_split["test"])

        self.assertEqual(len(train_files & val_files), 0, "Duplicate files between train and val")
        self.assertEqual(len(train_files & test_files), 0, "Duplicate files between train and test")
        self.assertEqual(len(val_files & test_files), 0, "Duplicate files between val and test")

    def test_total_image_counts(self):
        """Verify total image counts match expected clean dataset size."""
        n_tr = len(self.files_by_split["train"])
        n_val = len(self.files_by_split["val"])
        n_te = len(self.files_by_split["test"])
        total = n_tr + n_val + n_te

        self.assertEqual(total, 10292, f"Expected 10,292 clean images, got {total}")
        self.assertEqual(n_tr, 7158, f"Expected 7,158 train images, got {n_tr}")
        self.assertEqual(n_val, 1546, f"Expected 1,546 val images, got {n_val}")
        self.assertEqual(n_te, 1588, f"Expected 1,588 test images, got {n_te}")

    def test_sanitization_report_validity(self):
        """Verify sanitization_report.json exists, is valid JSON, and matches disk numbers."""
        self.assertTrue(self.report_path.exists(), "sanitization_report.json missing!")
        with open(self.report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        self.assertIn("metadata", report)
        self.assertIn("dataset_summary", report)
        self.assertIn("partition_totals", report)
        self.assertIn("zero_leakage_audit", report)
        self.assertIn("per_class_breakdown", report)

        self.assertTrue(report["zero_leakage_audit"]["zero_leakage_verified"])
        self.assertEqual(report["zero_leakage_audit"]["train_val_base_overlap"], 0)
        self.assertEqual(report["zero_leakage_audit"]["train_test_base_overlap"], 0)
        self.assertEqual(report["zero_leakage_audit"]["val_test_base_overlap"], 0)

        # Check report counts match disk counts
        self.assertEqual(report["partition_totals"]["train"]["image_count"], len(self.files_by_split["train"]))
        self.assertEqual(report["partition_totals"]["val"]["image_count"], len(self.files_by_split["val"]))
        self.assertEqual(report["partition_totals"]["test"]["image_count"], len(self.files_by_split["test"]))

    def test_sample_image_readability(self):
        """Verify that images in clean dataset can be opened and read without corruption."""
        for s in self.splits:
            sample_files = self.files_by_split[s][:10]
            for cname, fname, path in sample_files:
                with Image.open(path) as img:
                    width, height = img.size
                    self.assertGreater(width, 0)
                    self.assertGreater(height, 0)

    def test_base_id_canonicalization_edge_cases(self):
        """Verify canonical extraction maps all known filename oddities to the correct base leaf ID."""
        test_cases = [
            ("Blueberry leaf (3.jpg", "Blueberry leaf (3)"),
            ("Blueberry leaf (3).jpg", "Blueberry leaf (3)"),
            ("Blueberry leaf (3)_aug_p1_46.jpg", "Blueberry leaf (3)"),
            ("Apple leaf (53)_aug.jpg", "Apple leaf (53)"),
            ("Peach leaf (10)AUG_1.jpg", "Peach leaf (10)"),
            ("Peach leaf (10)AUG_2.jpg", "Peach leaf (10)"),
            ("grape leaf (24)Aug_1.jpg", "grape leaf (24)"),
            ("Peach leaf (18)1.jpg", "Peach leaf (18)"),
            ("Peach leaf (18)2.jpg", "Peach leaf (18)"),
            ("Tomato leaf (3)1.jpg", "Tomato leaf (3)"),
            ("Tomato leaf (3)2.jpg", "Tomato leaf (3)"),
            ("Strawberry leaf (82.jpg", "Strawberry leaf (82)"),
            ("Strawberry leaf (82)1.jpg", "Strawberry leaf (82)"),
            ("Tomato leaf late blight (63.jpg", "Tomato leaf late blight (63)"),
            ("Tomato leaf late blight (63).jpg", "Tomato leaf late blight (63)"),
            ("Apple Scab Leaf (17)_aug_p2_62(2).jpg", "Apple Scab Leaf (17)"),
            ("Potato leaf late blight (73)_aug_p0_1.jpg", "Potato leaf late blight (73)"),
        ]
        for fname, expected in test_cases:
            actual = extract_base_id(fname)
            self.assertEqual(actual, expected, f"Failed for {fname}: expected {expected}, got {actual}")


if __name__ == "__main__":
    unittest.main()
