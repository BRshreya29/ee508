#!/usr/bin/env python3
"""Step 4, Stream 2 — Extract frame-difference (motion) features.

Reference: doc 03 — Frame Difference.
- Grayscale frame difference + adaptive avg pool to 7x7 = 49d
- Frame 0 = zero vector
- Output per video: features/{video_id}/framediff.pt — shape [32, 49]
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

POOL_SIZE = (7, 7)
D_DIFF = POOL_SIZE[0] * POOL_SIZE[1]  # 49


def compute_frame_diff(frame_t, frame_prev, pool_size=POOL_SIZE):
    """Compute absolute grayscale difference with spatial pooling.

    Args:
        frame_t: numpy array HxWx3, uint8
        frame_prev: numpy array HxWx3, uint8
        pool_size: spatial pooling grid size

    Returns:
        1D numpy array of size pool_size[0] * pool_size[1]
    """
    import cv2

    gray_t = cv2.cvtColor(frame_t, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY).astype(np.float32)

    diff = np.abs(gray_t - gray_prev)  # [H, W], range [0, 255]
    diff_normalized = diff / 255.0  # normalize to [0, 1]

    # Spatial average pooling to fixed grid
    diff_tensor = torch.tensor(diff_normalized).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    pooled = F.adaptive_avg_pool2d(diff_tensor, pool_size)  # [1, 1, 7, 7]
    return pooled.squeeze().flatten().numpy()  # [49]


def extract_framediff_features(video_path, frame_indices):
    """Extract frame-difference features for all 32 sampled frames.

    Args:
        video_path: path to video file
        frame_indices: list of 32 integer frame indices

    Returns:
        tensor of shape [32, 49]
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Read all needed frames
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            if frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        else:
            frames.append(frame)
    cap.release()

    # Compute frame differences
    features = []
    for t in range(len(frames)):
        if t == 0:
            feat = np.zeros(D_DIFF, dtype=np.float32)
        else:
            feat = compute_frame_diff(frames[t], frames[t - 1])
        features.append(feat)

    return torch.tensor(np.stack(features))  # [32, 49]


def main():
    parser = argparse.ArgumentParser(description="Extract frame difference features")
    parser.add_argument("--video_root", type=str, default="data/charades_videos")
    parser.add_argument("--frame_indices", type=str, default="data/frame_indices.json")
    parser.add_argument("--out_root", type=str, default="features")
    args = parser.parse_args()

    with open(args.frame_indices, "r") as f:
        frame_indices = json.load(f)

    for vid, indices in tqdm(frame_indices.items(), desc="Frame diff"):
        out_dir = os.path.join(args.out_root, vid)
        out_path = os.path.join(out_dir, "framediff.pt")
        if os.path.exists(out_path):
            continue

        video_path = os.path.join(args.video_root, f"{vid}.mp4")
        if not os.path.exists(video_path):
            video_path = os.path.join(args.video_root, "Charades_v1_480", f"{vid}.mp4")
        if not os.path.exists(video_path):
            video_path = os.path.join(args.video_root, f"{vid}.avi")
        if not os.path.exists(video_path):
            print(f"  Skipping {vid}: video not found")
            continue

        diff_feats = extract_framediff_features(video_path, indices)

        os.makedirs(out_dir, exist_ok=True)
        torch.save(diff_feats, out_path)

    print("Frame difference feature extraction complete.")


if __name__ == "__main__":
    main()
