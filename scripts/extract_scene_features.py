#!/usr/bin/env python3
"""Step 4, Stream 4 — Extract scene classifier features.

Reference: doc 05 — Scene Classifier.
- Loads CLIP CLS tokens from clip.pt (already extracted in Stream 1)
- Passes through trained SceneClassifier MLP
- Output per video: features/{video_id}/scene.pt — shape [32, 6]
"""
import os
import json
import sys
import argparse
import torch
from tqdm import tqdm

# Add src to path for SceneClassifier import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.scene_classifier import SceneClassifier


def main():
    parser = argparse.ArgumentParser(description="Extract scene classification features")
    parser.add_argument("--features_root", type=str, default="features",
                        help="Root directory with per-video feature subdirs")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/scene_mlp.pt",
                        help="Path to trained SceneClassifier checkpoint")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load scene classifier
    scene_classifier = SceneClassifier(input_dim=512, num_classes=6)
    if os.path.exists(args.checkpoint):
        scene_classifier.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded scene classifier from {args.checkpoint}")
    else:
        print(f"WARNING: No checkpoint found at {args.checkpoint}")
        print("Using randomly initialized scene classifier (for development only).")

    scene_classifier.eval()
    for param in scene_classifier.parameters():
        param.requires_grad = False
    scene_classifier = scene_classifier.to(device)

    # Find all videos with clip.pt
    video_ids = []
    for name in sorted(os.listdir(args.features_root)):
        clip_path = os.path.join(args.features_root, name, "clip.pt")
        if os.path.exists(clip_path):
            video_ids.append(name)

    print(f"Processing {len(video_ids)} videos...")

    for vid in tqdm(video_ids, desc="Scene features"):
        clip_path = os.path.join(args.features_root, vid, "clip.pt")
        out_path = os.path.join(args.features_root, vid, "scene.pt")

        if os.path.exists(out_path):
            continue

        clip_feats = torch.load(clip_path, map_location=device)  # [32, 512]

        with torch.no_grad():
            logits = scene_classifier(clip_feats)  # [32, 6]
            scene_probs = logits.softmax(dim=-1)  # [32, 6]

        torch.save(scene_probs.cpu(), out_path)

    print("Scene feature extraction complete.")


if __name__ == "__main__":
    main()
