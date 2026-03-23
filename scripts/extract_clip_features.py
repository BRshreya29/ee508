#!/usr/bin/env python3
"""Step 4, Stream 1+4 — Extract CLIP ViT-B/16 CLS tokens.

Reference: doc 02 — Video Backbone.
- Frozen CLIP ViT-B/16 from HuggingFace
- Output per video: features/{video_id}/clip.pt — shape [32, 512]
- The same CLS tokens are reused by Stream 4 (scene classifier)
"""
import os
import json
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_clip_model(device="cpu"):
    """Load frozen CLIP ViT-B/16."""
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)
    return model, processor


def extract_frames_from_video(video_path, frame_indices):
    """Extract specific frames from a video file as PIL images."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            # Fallback: use last successful frame or black frame
            if frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(Image.new("RGB", (224, 224)))
            continue
        # Convert BGR to RGB PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames


def extract_clip_tokens(model, processor, frames, device="cpu", batch_size=8):
    """Extract CLS tokens from CLIP for a list of frames.

    Args:
        model: CLIP model
        processor: CLIP processor
        frames: list of PIL images (32 frames)
        device: torch device
        batch_size: frames per batch

    Returns:
        tensor of shape [32, 512]
    """
    all_tokens = []
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        inputs = processor(images=batch_frames, return_tensors="pt", padding=True)
        # Move pixel_values to device
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            vision_outputs = model.vision_model(pixel_values=pixel_values)
        cls_tokens = vision_outputs.last_hidden_state[:, 0, :]  # [batch, 768]
        # CLIP ViT-B/16 has hidden_size=768, but pooler output is 512
        cls_tokens = vision_outputs.pooler_output  # [batch, 512]
        all_tokens.append(cls_tokens.cpu())

    return torch.cat(all_tokens, dim=0)  # [32, 512]


def main():
    parser = argparse.ArgumentParser(description="Extract CLIP CLS tokens")
    parser.add_argument("--video_root", type=str, default="data/charades_videos")
    parser.add_argument("--frame_indices", type=str, default="data/frame_indices.json")
    parser.add_argument("--out_root", type=str, default="features")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    with open(args.frame_indices, "r") as f:
        frame_indices = json.load(f)

    device = torch.device(args.device)
    print(f"Loading CLIP ViT-B/16 on {device}...")
    model, processor = load_clip_model(device)

    for vid, indices in tqdm(frame_indices.items(), desc="Extracting CLIP"):
        out_dir = os.path.join(args.out_root, vid)
        out_path = os.path.join(out_dir, "clip.pt")
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

        frames = extract_frames_from_video(video_path, indices)
        cls_tokens = extract_clip_tokens(model, processor, frames, device, args.batch_size)

        os.makedirs(out_dir, exist_ok=True)
        torch.save(cls_tokens, out_path)

    print("CLIP feature extraction complete.")


if __name__ == "__main__":
    main()
