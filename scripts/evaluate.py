#!/usr/bin/env python3
"""Evaluate trained model on the test split.

Computes:
  - per-class AP (Average Precision)
  - overall mAP
  - Precision, Recall, F1 at threshold 0.5
  - per-class breakdown for the top 20 most common classes
"""
import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import SceneActivityModel
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score

CLASSES_FILE = "data/Charades_v1_classes.txt"


def load_class_mapping(filepath):
    mapping = {}
    if os.path.exists(filepath):
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2 and parts[0].startswith("c"):
                    try:
                        mapping[int(parts[0][1:])] = parts[1]
                    except ValueError:
                        pass
    return mapping


def load_features(features_dir):
    def load(name):
        return torch.load(os.path.join(features_dir, name), weights_only=True)

    clip_feat = load("clip.pt")
    diff_feat = load("framediff.pt")
    obj_feat = load("objects.pt")
    scene_feat = load("scene.pt")
    obj_mask_path = os.path.join(features_dir, "obj_mask.pt")
    if os.path.exists(obj_mask_path):
        obj_mask = torch.load(obj_mask_path, weights_only=True)
    else:
        obj_mask = (obj_feat.abs().sum(dim=-1) == 0)

    return (
        clip_feat.unsqueeze(0),
        diff_feat.unsqueeze(0),
        obj_feat.unsqueeze(0),
        scene_feat.unsqueeze(0),
        obj_mask.unsqueeze(0),
    )


def main():
    features_dir = "features"
    checkpoint = "checkpoints/best_model.pt"
    num_classes = 157
    threshold = 0.5

    with open("data/split.json") as f:
        split = json.load(f)
    with open("data/labels.json") as f:
        labels = json.load(f)

    test_ids = split["test"]
    cls_map = load_class_mapping(CLASSES_FILE)

    model = SceneActivityModel()
    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded model from {checkpoint}")
    print(f"Evaluating on {len(test_ids)} test videos...\n")

    all_preds = []
    all_targets = []
    skipped = 0

    for vid in test_ids:
        vid_dir = os.path.join(features_dir, vid)
        if not os.path.exists(os.path.join(vid_dir, "clip.pt")):
            skipped += 1
            continue

        label_entry = labels.get(vid, {})
        activity_indices = label_entry.get("activity", [])

        target = np.zeros(num_classes, dtype=np.float32)
        for idx in activity_indices:
            if 0 <= idx < num_classes:
                target[idx] = 1.0

        try:
            clip_feat, diff_feat, obj_feat, scene_feat, obj_mask = load_features(vid_dir)
            with torch.no_grad():
                act_logits, _, _, _ = model(clip_feat, diff_feat, obj_feat, scene_feat, obj_mask)
            pred = act_logits[0].sigmoid().numpy()
        except Exception as e:
            skipped += 1
            continue

        all_preds.append(pred)
        all_targets.append(target)

    if skipped > 0:
        print(f"Note: skipped {skipped} videos (no features on disk)\n")

    all_preds = np.stack(all_preds)    # [N, 157]
    all_targets = np.stack(all_targets)  # [N, 157]
    N = len(all_preds)
    print(f"Evaluated on {N} videos.\n")

    # ── Overall Metrics ───────────────────────────────────────────────
    # mAP (only for classes that have at least 1 positive example)
    aps = []
    for c in range(num_classes):
        if all_targets[:, c].sum() > 0:
            aps.append(average_precision_score(all_targets[:, c], all_preds[:, c]))
    mAP = np.mean(aps) if aps else 0.0

    bin_preds = (all_preds >= threshold).astype(np.float32)
    precision = precision_score(all_targets, bin_preds, average="micro", zero_division=0)
    recall = recall_score(all_targets, bin_preds, average="micro", zero_division=0)
    f1 = f1_score(all_targets, bin_preds, average="micro", zero_division=0)

    print("=" * 60)
    print(f"  Overall mAP            : {mAP:.4f}  ({mAP*100:.2f}%)")
    print(f"  Precision  (t={threshold}):  {precision:.4f}  ({precision*100:.2f}%)")
    print(f"  Recall     (t={threshold}):  {recall:.4f}  ({recall*100:.2f}%)")
    print(f"  F1-score   (t={threshold}):  {f1:.4f}  ({f1*100:.2f}%)")
    print("=" * 60)

    # ── Per-class breakdown (top 20 most common in test set) ──────────
    class_counts = all_targets.sum(axis=0)
    top_classes = np.argsort(-class_counts)[:20]

    print("\nPer-class breakdown (top 20 most-seen test classes):")
    print(f"{'Class':<45} {'AP':>6}  {'P':>6}  {'R':>6}  {'F1':>6}  {'N':>5}")
    print("-" * 80)
    for c in top_classes:
        n_pos = int(class_counts[c])
        if n_pos == 0:
            continue
        ap = average_precision_score(all_targets[:, c], all_preds[:, c])
        p  = precision_score(all_targets[:, c], bin_preds[:, c], zero_division=0)
        r  = recall_score(all_targets[:, c], bin_preds[:, c], zero_division=0)
        f  = f1_score(all_targets[:, c], bin_preds[:, c], zero_division=0)
        name = cls_map.get(c, f"Class {c}")[:43]
        print(f"{name:<45} {ap:>6.3f}  {p:>6.3f}  {r:>6.3f}  {f:>6.3f}  {n_pos:>5}")


if __name__ == "__main__":
    main()
