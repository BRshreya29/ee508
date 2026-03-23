#!/usr/bin/env python3
"""Step 4, Stream 3 — Extract DETR object detection features.

Reference: doc 04 — Object Detector.
- Frozen DETR (facebook/detr-resnet-50)
- Top K=5 detections by confidence
- Each token: concat(class_one_hot[64], bbox[4], detr_feat[256]) = 324d
- Output per video: features/{video_id}/objects.pt — shape [32, 5, 324]
"""
import os
import json
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

K = 5  # top-K detections per frame
CLASS_EMBED_DIM = 64
BBOX_DIM = 4
DETR_FEAT_DIM = 256
TOKEN_DIM = CLASS_EMBED_DIM + BBOX_DIM + DETR_FEAT_DIM  # 324

# Working taxonomy — 8 object classes (doc 04)
OBJECT_CLASSES = ["person", "chair", "table", "cup", "phone", "book", "laptop", "door"]
NUM_OBJ_CLASSES = len(OBJECT_CLASSES)

# COCO class names → working taxonomy mapping (doc 04)
COCO_TO_LOCAL = {
    "person": "person",
    "chair": "chair",
    "dining table": "table",
    "cup": "cup",
    "cell phone": "phone",
    "book": "book",
    "laptop": "laptop",
    # "door" is not a COCO class — detections not matching are treated as padding
}

# Build COCO label IDs → local mapping at load time
LOCAL_CLASS_TO_IDX = {c: i for i, c in enumerate(OBJECT_CLASSES)}


def load_detr_model(device="cpu"):
    """Load frozen DETR model."""
    from transformers import DetrForObjectDetection, DetrImageProcessor

    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)
    return model, processor


def get_coco_label_names(model):
    """Get COCO class names from DETR's id2label."""
    return model.config.id2label


def extract_object_tokens(model, processor, frame_pil, coco_id2label, device="cpu"):
    """Extract top-K object tokens from a single frame.

    Returns:
        tokens: tensor [K, 324]
        is_pad: tensor [K] — True where slot is padding
    """
    inputs = processor(images=frame_pil, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, output_hidden_states=True)

    logits = outputs.logits[0]  # [num_queries, num_classes + 1]
    bboxes = outputs.pred_boxes[0]  # [num_queries, 4] (cx, cy, w, h) normalized

    # Get DETR decoder features
    if hasattr(outputs, "decoder_hidden_states") and outputs.decoder_hidden_states is not None:
        detr_feats = outputs.decoder_hidden_states[-1][0]  # [num_queries, 256]
    else:
        # Fallback: use last hidden state
        detr_feats = outputs.logits.new_zeros(logits.shape[0], DETR_FEAT_DIM)

    # Scores: max class prob excluding background (last class)
    probs = logits.softmax(-1)[:, :-1]
    scores, class_ids = probs.max(-1)

    # Select top K by confidence
    topk_k = min(K, scores.shape[0])
    topk = scores.topk(topk_k)
    selected_indices = topk.indices

    tokens = []
    is_pad = []
    for idx in selected_indices:
        coco_class_id = class_ids[idx].item()
        coco_name = coco_id2label.get(coco_class_id, "")

        # Map COCO class to local taxonomy
        local_name = COCO_TO_LOCAL.get(coco_name, None)

        if local_name is not None:
            local_idx = LOCAL_CLASS_TO_IDX[local_name]
            # One-hot class embedding (64d with one hot in first NUM_OBJ_CLASSES positions)
            class_embed = torch.zeros(CLASS_EMBED_DIM)
            class_embed[local_idx] = 1.0
            is_pad.append(False)
        else:
            # Not in working taxonomy → treated as padding
            class_embed = torch.zeros(CLASS_EMBED_DIM)
            is_pad.append(True)

        feat = torch.cat([
            class_embed,
            bboxes[idx].cpu(),  # [4]
            detr_feats[idx].cpu()[:DETR_FEAT_DIM],  # [256]
        ])  # [324]
        tokens.append(feat)

    # Pad to K if fewer detections
    while len(tokens) < K:
        tokens.append(torch.zeros(TOKEN_DIM))
        is_pad.append(True)

    return torch.stack(tokens[:K]), torch.tensor(is_pad[:K])


def main():
    parser = argparse.ArgumentParser(description="Extract DETR object features")
    parser.add_argument("--video_root", type=str, default="data/charades_videos")
    parser.add_argument("--frame_indices", type=str, default="data/frame_indices.json")
    parser.add_argument("--out_root", type=str, default="features")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    with open(args.frame_indices, "r") as f:
        frame_indices = json.load(f)

    device = torch.device(args.device)
    print(f"Loading DETR on {device}...")
    model, processor = load_detr_model(device)
    coco_id2label = get_coco_label_names(model)

    for vid, indices in tqdm(frame_indices.items(), desc="DETR objects"):
        out_dir = os.path.join(args.out_root, vid)
        out_path = os.path.join(out_dir, "objects.pt")
        if os.path.exists(out_path):
            continue

        video_path = os.path.join(args.video_root, f"{vid}.mp4")
        if not os.path.exists(video_path):
            video_path = os.path.join(args.video_root, f"{vid}.avi")
        if not os.path.exists(video_path):
            print(f"  Skipping {vid}: video not found")
            continue

        import cv2
        cap = cv2.VideoCapture(video_path)
        all_tokens = []
        all_masks = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                # Black frame fallback
                frame_pil = Image.new("RGB", (224, 224))
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)

            tokens, is_pad = extract_object_tokens(
                model, processor, frame_pil, coco_id2label, device
            )
            all_tokens.append(tokens)
            all_masks.append(is_pad)
        cap.release()

        obj_features = torch.stack(all_tokens)  # [32, 5, 324]
        obj_masks = torch.stack(all_masks)  # [32, 5]

        os.makedirs(out_dir, exist_ok=True)
        torch.save(obj_features, out_path)
        torch.save(obj_masks, os.path.join(out_dir, "obj_mask.pt"))

    print("DETR object feature extraction complete.")


if __name__ == "__main__":
    main()
