"""
Unit and Integration Tests for DINOv2-Large LoRA + MAF Training Pipeline
Verifies architecture, pooling mathematics, scheduler curves, and checkpointing logic.
"""

import sys
import unittest
import math
import tempfile
import shutil
from pathlib import Path

import torch
import torch.nn as nn

# Include workspace root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_dinov2_lora import (
    MultiHeadAttentivePooling,
    DinoV2LoRAPathologyModel,
    build_lr_scheduler,
    get_train_transforms,
    get_val_transforms,
    PlantDocCleanDataset,
)
from utils.checkpoint_manager import CheckpointManager


class TestDinoV2Pipeline(unittest.TestCase):

    def test_multi_head_attentive_pooling_shapes_and_attention(self):
        """Verify MAF pooling shapes, multi-head projection, and attention matrix properties."""
        batch_size = 4
        num_patches = 256
        embed_dim = 1024
        num_heads = 8

        maf = MultiHeadAttentivePooling(embed_dim=embed_dim, num_heads=num_heads, dropout=0.0)
        maf.eval()

        dummy_patches = torch.randn(batch_size, num_patches, embed_dim)
        pooled, attn = maf(dummy_patches, return_attention=True)

        # Output shape must be (B, embed_dim)
        self.assertEqual(pooled.shape, (batch_size, embed_dim))

        # Attention shape must be (B, num_heads, num_patches)
        self.assertEqual(attn.shape, (batch_size, num_heads, num_patches))

        # Attention weights along patch dimension must sum to 1.0 (valid probability distribution)
        attn_sums = attn.sum(dim=-1)
        torch.testing.assert_close(attn_sums, torch.ones_like(attn_sums), rtol=1e-4, atol=1e-4)

    def test_maf_gradient_flow(self):
        """Verify that gradients propagate properly through MAF pooling to learned query."""
        embed_dim = 1024
        num_heads = 8
        maf = MultiHeadAttentivePooling(embed_dim=embed_dim, num_heads=num_heads, dropout=0.0)
        maf.train()

        dummy_patches = torch.randn(2, 256, embed_dim, requires_grad=True)
        pooled, _ = maf(dummy_patches, return_attention=False)
        loss = (pooled ** 2).sum()
        loss.backward()

        self.assertIsNotNone(maf.query.grad)
        self.assertGreater(maf.query.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(dummy_patches.grad)

    def test_lr_scheduler_warmup_and_cosine_curve(self):
        """Verify 3-epoch linear warmup (1e-6 to 2e-4) and 20-epoch cosine annealing (down to 1e-6)."""
        dummy_param = nn.Parameter(torch.zeros(10))
        lr_max = 2e-4
        lr_min = 1e-6
        optimizer = torch.optim.AdamW([dummy_param], lr=lr_max)

        steps_per_epoch = 100
        warmup_epochs = 3
        total_epochs = 23
        total_steps = steps_per_epoch * total_epochs
        warmup_steps = steps_per_epoch * warmup_epochs

        scheduler = build_lr_scheduler(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            lr_min=lr_min,
            lr_max=lr_max,
        )

        # Step 0: Initial LR should be lr_min
        lrs = []
        for step in range(total_steps):
            current_lr = optimizer.param_groups[0]["lr"]
            lrs.append(current_lr)
            optimizer.step()
            scheduler.step()

        # Check step 0
        self.assertAlmostEqual(lrs[0], lr_min, delta=1e-7)

        # Check at end of warmup (step = warmup_steps)
        self.assertAlmostEqual(lrs[warmup_steps], lr_max, delta=1e-6)

        # Check midpoint of cosine decay
        mid_decay_step = warmup_steps + (total_steps - warmup_steps) // 2
        expected_mid = lr_min + 0.5 * (lr_max - lr_min)
        self.assertAlmostEqual(lrs[mid_decay_step], expected_mid, delta=5e-6)

        # Check end of cosine decay (final step)
        self.assertAlmostEqual(lrs[-1], lr_min, delta=1e-6)

    def test_dataset_transforms_and_loading(self):
        """Verify Albumentations transform produces normalized (3, 224, 224) tensors."""
        clean_dir = Path(r"c:\CropGuard v3\augmented_plantdoc_clean\val")
        if not clean_dir.exists():
            self.skipTest("Clean val directory not found")

        val_transform = get_val_transforms(image_size=224)
        ds = PlantDocCleanDataset(clean_dir, transform=val_transform)

        self.assertGreater(len(ds), 0)
        self.assertEqual(len(ds.classes), 27)

        tensor, label = ds[0]
        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertIsInstance(label, int)
        self.assertGreaterEqual(label, 0)
        self.assertLess(label, 27)

    def test_checkpoint_peak_saving_logic(self):
        """Verify atomic CheckpointManager saves latest every epoch and updates best on peak."""
        temp_dir = Path(tempfile.mkdtemp(prefix="test_ckpt_"))
        try:
            mgr = CheckpointManager(checkpoint_dir=str(temp_dir), experiment_name="test_peak")
            
            dummy_model = nn.Linear(10, 2)
            dummy_opt = torch.optim.SGD(dummy_model.parameters(), lr=0.01)

            # Epoch 1: F1 = 0.50 -> Should be best
            mgr.save_checkpoint(epoch=1, model=dummy_model, optimizer=dummy_opt, val_metric=0.50, is_best=True)
            self.assertTrue(mgr.latest_path.exists())
            self.assertTrue(mgr.best_path.exists())
            ckpt_best = torch.load(mgr.best_path, map_location="cpu")
            self.assertEqual(ckpt_best["val_metric"], 0.50)
            self.assertEqual(ckpt_best["epoch"], 1)

            # Epoch 2: F1 = 0.45 (Not peak) -> Latest updated, best unchanged
            mgr.save_checkpoint(epoch=2, model=dummy_model, optimizer=dummy_opt, val_metric=0.45, is_best=False)
            ckpt_latest = torch.load(mgr.latest_path, map_location="cpu")
            ckpt_best = torch.load(mgr.best_path, map_location="cpu")
            self.assertEqual(ckpt_latest["epoch"], 2)
            self.assertEqual(ckpt_latest["val_metric"], 0.45)
            # Best should still be Epoch 1!
            self.assertEqual(ckpt_best["epoch"], 1)
            self.assertEqual(ckpt_best["val_metric"], 0.50)

            # Epoch 3: F1 = 0.75 (New peak!) -> Latest updated, best updated to Epoch 3
            mgr.save_checkpoint(epoch=3, model=dummy_model, optimizer=dummy_opt, val_metric=0.75, is_best=True)
            ckpt_best = torch.load(mgr.best_path, map_location="cpu")
            self.assertEqual(ckpt_best["epoch"], 3)
            self.assertEqual(ckpt_best["val_metric"], 0.75)
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_dinov2_lora_pathology_model_forward_and_params(self):
        """Verify DINOv2 LoRA model initialization, frozen backbone, trainable parameter count, and forward shapes."""
        model = DinoV2LoRAPathologyModel(
            backbone_id="facebook/dinov2-large",
            num_classes=27,
            lora_r=16,
            lora_alpha=32,
            pooling_heads=8,
        )
        model.eval()

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        # Frozen backbone should keep total params around ~300M+ and trainable params under 10M (~5.5M)
        self.assertGreater(total_params, 300_000_000)
        self.assertLess(trainable_params, 10_000_000)
        self.assertGreater(trainable_params, 1_000_000)

        # Base backbone parameters must be frozen
        base_layer = model.backbone.base_model.model.embeddings.patch_embeddings.projection
        self.assertFalse(base_layer.weight.requires_grad)

        # Forward pass with dummy image batch
        dummy_input = torch.randn(2, 3, 224, 224)
        logits, attn = model(dummy_input, return_attention=True)

        self.assertEqual(logits.shape, (2, 27))
        self.assertEqual(attn.shape, (2, 8, 256))


if __name__ == "__main__":
    unittest.main()
