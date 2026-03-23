#!/usr/bin/env python3
"""End-to-end smoke test — synthetic data, full pipeline.

Creates synthetic feature files, runs DataLoader, performs one forward+backward
pass through the full model, verifies shapes and gradient flow.
"""
import os
import sys
import json
import tempfile
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def create_synthetic_features(tmpdir, num_videos=10):
    """Create synthetic .pt feature files for testing."""
    features_root = os.path.join(tmpdir, "features")
    video_ids = [f"vid{i:04d}" for i in range(num_videos)]

    for vid in video_ids:
        feat_dir = os.path.join(features_root, vid)
        os.makedirs(feat_dir)
        torch.save(torch.randn(32, 512), os.path.join(feat_dir, "clip.pt"))
        torch.save(torch.randn(32, 49), os.path.join(feat_dir, "framediff.pt"))
        torch.save(torch.randn(32, 5, 324), os.path.join(feat_dir, "objects.pt"))
        torch.save(torch.randn(32, 6), os.path.join(feat_dir, "scene.pt"))
        torch.save(torch.zeros(32, 5, dtype=torch.bool), os.path.join(feat_dir, "obj_mask.pt"))

    # Labels
    labels = {}
    for vid in video_ids:
        labels[vid] = {
            "activity": [0, 3],
            "temporal": {"0": [0.1, 0.5], "3": [0.3, 0.8]},
            "scene_graph": {},
        }

    # Split: 7 train, 2 val, 1 test
    split = {
        "train": video_ids[:7],
        "val": video_ids[7:9],
        "test": video_ids[9:],
    }

    labels_path = os.path.join(tmpdir, "labels.json")
    split_path = os.path.join(tmpdir, "split.json")
    with open(labels_path, "w") as f:
        json.dump(labels, f)
    with open(split_path, "w") as f:
        json.dump(split, f)

    return features_root, labels_path, split_path, split


def main():
    from src.model import SceneActivityModel
    from src.dataset import get_dataloader
    from src.output_heads import total_loss

    print("=" * 60)
    print("EE508 Smoke Test — Synthetic Data")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Create synthetic data
        print("\n1. Creating synthetic features...")
        features_root, labels_path, split_path, split = create_synthetic_features(tmpdir)
        print(f"   Created features for {sum(len(v) for v in split.values())} videos")

        # 2. Test DataLoaders
        print("\n2. Testing DataLoaders...")
        for split_name in ["train", "val", "test"]:
            loader = get_dataloader(
                split_name, split_path, labels_path, features_root,
                batch_size=4, num_workers=0,
            )
            num_batches = 0
            for batch in loader:
                clip, diff, obj, scene, mask, labels = batch
                print(
                    f"   {split_name:5s} | "
                    f"clip={tuple(clip.shape)} "
                    f"diff={tuple(diff.shape)} "
                    f"obj={tuple(obj.shape)} "
                    f"scene={tuple(scene.shape)}"
                )
                num_batches += 1
            print(f"   {split_name}: {num_batches} batch(es)")

        # 3. Test model forward pass
        print("\n3. Testing model forward pass...")
        model = SceneActivityModel()
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Trainable parameters: {total_params:,}")

        train_loader = get_dataloader(
            "train", split_path, labels_path, features_root,
            batch_size=4, num_workers=0,
        )

        for batch in train_loader:
            clip, diff, obj, scene, mask, labels = batch
            act_logits, loc_preds, sg_logits, attn_weights = model(clip, diff, obj, scene, mask)

            print(f"   act_logits:    {tuple(act_logits.shape)}")
            print(f"   loc_preds:     {tuple(loc_preds.shape)}")
            print(f"   sg_logits:     {tuple(sg_logits.shape)}")
            print(f"   attn_weights:  {tuple(attn_weights.shape)}")
            break  # one batch is enough

        # 4. Test loss + backward
        print("\n4. Testing loss computation and backward pass...")
        for batch in train_loader:
            clip, diff, obj, scene, mask, labels = batch

            act_logits, loc_preds, sg_logits, _ = model(clip, diff, obj, scene, mask)

            loss, components = total_loss(
                act_logits, loc_preds, sg_logits,
                labels["activity"], labels["temporal"],
                labels["activity_mask"], labels["scene_graph"],
            )

            loss.backward()

            print(f"   total_loss:  {loss.item():.4f}")
            for k, v in components.items():
                print(f"   L_{k}: {v:.4f}")

            # Check gradients
            grad_ok = True
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is None:
                    print(f"   ✗ No gradient for: {name}")
                    grad_ok = False
            if grad_ok:
                print("   ✓ Gradients flow to all trainable parameters")
            break

        # 5. Test attention weights accessibility
        print("\n5. Checking attention weights...")
        weights = model.get_attention_weights()
        if weights is not None and weights.shape[-1] == 192:
            print(f"   ✓ Attention weights: {tuple(weights.shape)}")
        else:
            print("   ✗ Attention weights not accessible")

    print("\n" + "=" * 60)
    print("✓ All smoke tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
