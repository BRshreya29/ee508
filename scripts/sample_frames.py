#!/usr/bin/env python3
"""Step 3 — Uniformly sample N=32 frames per video.

Reference: doc 01 Step 3.
- np.linspace(0, total_frames-1, 32, dtype=int) per video
- Videos shorter than 32 frames: repeat last frame to pad
"""
import os
import json
import argparse
import numpy as np

NUM_FRAMES = 32


def sample_frame_indices(total_frames, n=NUM_FRAMES):
    """Compute N uniformly-spaced frame indices.

    Args:
        total_frames: total number of frames in the video
        n: number of frames to sample (default 32)

    Returns:
        numpy array of shape [n] with integer frame indices
    """
    if total_frames <= 0:
        return np.zeros(n, dtype=int)

    if total_frames < n:
        # Pad by repeating last frame
        indices = np.arange(total_frames, dtype=int)
        pad_count = n - total_frames
        indices = np.concatenate([indices, np.full(pad_count, total_frames - 1, dtype=int)])
        return indices

    return np.linspace(0, total_frames - 1, n, dtype=int)


def get_video_frame_count(video_path):
    """Get total frame count from a video file using OpenCV."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def main():
    parser = argparse.ArgumentParser(description="Sample 32 frames per video")
    parser.add_argument("--video_root", type=str, default="data/charades_videos",
                        help="Directory containing video files")
    parser.add_argument("--annotations", type=str, default="data/filtered_annotations.json",
                        help="Filtered annotation file (for video ID list)")
    parser.add_argument("--out", type=str, default="data/frame_indices.json",
                        help="Output frame indices file")
    args = parser.parse_args()

    # Load video IDs from annotations
    with open(args.annotations, "r") as f:
        annotations = json.load(f)

    video_ids = list(annotations.keys())
    print(f"Processing {len(video_ids)} videos...")

    frame_indices = {}
    for i, vid in enumerate(video_ids):
        video_path = os.path.join(args.video_root, f"{vid}.mp4")
        if not os.path.exists(video_path):
            video_path = os.path.join(args.video_root, "Charades_v1_480", f"{vid}.mp4")
        if not os.path.exists(video_path):
            # Try .avi
            video_path = os.path.join(args.video_root, f"{vid}.avi")

        if not os.path.exists(video_path):
            print(f"  Warning: video not found for {vid}, skipping")
            continue

        total = get_video_frame_count(video_path)
        indices = sample_frame_indices(total, NUM_FRAMES)
        frame_indices[vid] = indices.tolist()

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(video_ids)} videos")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(frame_indices, f)

    print(f"Saved frame indices for {len(frame_indices)} videos to {args.out}")


if __name__ == "__main__":
    main()
