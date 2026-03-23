#!/usr/bin/env python3
"""Demo script — run inference on a video and produce 3 visualizations.

Reference: doc 00 — Demo (End of Project).
1. Activity timeline — bar chart with predicted activities + temporal extents
2. Scene graph — graph for selected frame showing object nodes + relation edges
3. Attention heatmap — 32×K cross-attention weight grid
"""
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))
from src.model import SceneActivityModel
from src.output_heads import RELATION_CLASSES, decode_segments, decode_triplets

OBJECT_CLASSES = ["person", "chair", "table", "cup", "phone", "book", "laptop", "door"]


def load_features(features_dir):
    """Load precomputed features for a single video."""
    clip_feat = torch.load(os.path.join(features_dir, "clip.pt"), weights_only=True)
    diff_feat = torch.load(os.path.join(features_dir, "framediff.pt"), weights_only=True)
    obj_feat = torch.load(os.path.join(features_dir, "objects.pt"), weights_only=True)
    scene_feat = torch.load(os.path.join(features_dir, "scene.pt"), weights_only=True)

    obj_mask_path = os.path.join(features_dir, "obj_mask.pt")
    if os.path.exists(obj_mask_path):
        obj_mask = torch.load(obj_mask_path, weights_only=True)
    else:
        obj_mask = (obj_feat.abs().sum(dim=-1) == 0)

    # Add batch dimension
    return (
        clip_feat.unsqueeze(0),
        diff_feat.unsqueeze(0),
        obj_feat.unsqueeze(0),
        scene_feat.unsqueeze(0),
        obj_mask.unsqueeze(0),
    )


def plot_activity_timeline(act_probs, loc_preds, out_path, threshold=0.3):
    """Visualization 1: Activity timeline bar chart."""
    act_probs_np = act_probs.numpy()
    loc_preds_np = loc_preds.numpy()  # [32, 2]

    active_classes = np.where(act_probs_np > threshold)[0]
    if len(active_classes) == 0:
        # Show top-5 classes by probability
        active_classes = np.argsort(act_probs_np)[-5:]

    fig, ax = plt.subplots(figsize=(14, max(3, len(active_classes) * 0.8)))

    colors = plt.cm.Set2(np.linspace(0, 1, len(active_classes)))

    for i, cls_idx in enumerate(active_classes):
        prob = act_probs_np[cls_idx]
        t_start = loc_preds_np[:, 0].mean()
        t_end = loc_preds_np[:, 1].mean()

        ax.barh(
            i, t_end - t_start, left=t_start,
            height=0.6, color=colors[i], alpha=0.8,
            edgecolor="black", linewidth=0.5,
        )
        ax.text(
            t_start + 0.01, i,
            f"Activity {cls_idx} (p={prob:.2f})",
            va="center", fontsize=9, fontweight="bold",
        )

    ax.set_xlim(0, 1)
    ax.set_xlabel("Normalized Time", fontsize=11)
    ax.set_ylabel("Activities", fontsize=11)
    ax.set_title("Activity Timeline", fontsize=13, fontweight="bold")
    ax.set_yticks(range(len(active_classes)))
    ax.set_yticklabels([f"Class {c}" for c in active_classes])
    ax.invert_yaxis()

    # Add frame markers at 32 positions
    frame_positions = np.linspace(0, 1, 32)
    for fp in frame_positions:
        ax.axvline(fp, color="gray", alpha=0.15, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved activity timeline: {out_path}")


def plot_scene_graph(sg_logits, obj_feat, frame_idx, out_path, threshold=0.5):
    """Visualization 2: Scene graph for a selected frame."""
    # sg_logits: [32, 20, 11]
    frame_logits = sg_logits[frame_idx]  # [20, 11]
    probs = frame_logits.softmax(dim=-1)

    # Get object class IDs from one-hot in first 64 dims of obj_feat
    obj_classes_frame = obj_feat[frame_idx, :, :8]  # [5, 8] — first 8 of 64d one-hot
    obj_class_ids = obj_classes_frame.argmax(dim=-1).numpy()  # [5]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Node positions in a circle
    K = 5
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    positions = [(0.5 + 0.35 * np.cos(a), 0.5 + 0.35 * np.sin(a)) for a in angles]

    # Draw nodes
    for i, (x, y) in enumerate(positions):
        cls_name = OBJECT_CLASSES[obj_class_ids[i]] if obj_class_ids[i] < len(OBJECT_CLASSES) else "unknown"
        circle = plt.Circle((x, y), 0.06, color="#4a90d9", alpha=0.8, ec="black", linewidth=1.5)
        ax.add_patch(circle)
        ax.text(x, y, cls_name, ha="center", va="center", fontsize=8, fontweight="bold", color="white")

    # Draw edges (relations)
    pair_idx = 0
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            best_rel = probs[pair_idx].argmax().item()
            rel_prob = probs[pair_idx, best_rel].item()

            if best_rel != 0 and rel_prob > threshold:
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                dx, dy = x2 - x1, y2 - y1

                ax.annotate(
                    "", xy=(x2 - 0.06 * dx / max(abs(dx) + abs(dy), 0.01), y2 - 0.06 * dy / max(abs(dx) + abs(dy), 0.01)),
                    xytext=(x1 + 0.06 * dx / max(abs(dx) + abs(dy), 0.01), y1 + 0.06 * dy / max(abs(dx) + abs(dy), 0.01)),
                    arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5),
                )
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                rel_name = RELATION_CLASSES[best_rel]
                ax.text(mid_x, mid_y + 0.02, rel_name, ha="center", va="center",
                        fontsize=7, color="#e74c3c", fontstyle="italic",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

            pair_idx += 1

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(f"Scene Graph — Frame {frame_idx}", fontsize=13, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved scene graph: {out_path}")


def plot_attention_heatmap(attn_weights, out_path):
    """Visualization 3: Cross-attention heatmap (32 × 32 frames).

    Shows which key frames each query temporal position attends to.
    """
    # attn_weights: [32, 192]
    # Object weights: positions 0-159, scene weights: 160-191
    obj_weights = attn_weights[:, :160].reshape(32, 32, 5)  # [Tq, Tk, K]
    # Sum over K objects per key frame
    heatmap = obj_weights.sum(dim=-1).numpy()  # [32, 32]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(heatmap, cmap="YlOrRd", aspect="auto", interpolation="nearest")
    ax.set_xlabel("Key Frame (object locations)", fontsize=11)
    ax.set_ylabel("Query Temporal Position", fontsize=11)
    ax.set_title("Cross-Attention: Temporal → Object Grounding", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Attention Weight")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved attention heatmap: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Demo: inference + visualization")
    parser.add_argument("--features_dir", type=str, required=True,
                        help="Directory with precomputed features for a video")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--out_dir", type=str, default="demo_output")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--frame_idx", type=int, default=16,
                        help="Frame index for scene graph visualization")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    model = SceneActivityModel()
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded model from {args.checkpoint}")
    else:
        print(f"WARNING: No checkpoint at {args.checkpoint}, using random weights")
    model = model.to(device)
    model.eval()

    # Load features
    clip_feat, diff_feat, obj_feat, scene_feat, obj_mask = load_features(args.features_dir)
    clip_feat = clip_feat.to(device)
    diff_feat = diff_feat.to(device)
    obj_feat = obj_feat.to(device)
    scene_feat = scene_feat.to(device)
    obj_mask = obj_mask.to(device)

    # Inference
    with torch.no_grad():
        act_logits, loc_preds, sg_logits, attn_weights = model(
            clip_feat, diff_feat, obj_feat, scene_feat, obj_mask
        )

    # Move to CPU for visualization
    act_probs = act_logits[0].cpu().sigmoid()
    loc_preds_cpu = loc_preds[0].cpu()
    sg_logits_cpu = sg_logits[0].cpu()
    attn_weights_cpu = attn_weights[0].cpu()
    obj_feat_cpu = obj_feat[0].cpu()

    # Generate all 3 visualizations
    plot_activity_timeline(
        act_probs, loc_preds_cpu,
        os.path.join(args.out_dir, "activity_timeline.png"),
    )
    plot_scene_graph(
        sg_logits_cpu, obj_feat_cpu, args.frame_idx,
        os.path.join(args.out_dir, "scene_graph.png"),
    )
    plot_attention_heatmap(
        attn_weights_cpu,
        os.path.join(args.out_dir, "attention_heatmap.png"),
    )

    print(f"\nAll visualizations saved to {args.out_dir}/")

    # Print summary
    print("\n--- Predictions ---")
    top_activities = act_probs.topk(5)
    print("Top-5 predicted activities:")
    for i in range(5):
        print(f"  Class {top_activities.indices[i].item()}: p={top_activities.values[i].item():.3f}")

    segments = decode_segments(loc_preds_cpu, act_probs)
    print(f"\nTemporal segments: {len(segments)} detected")
    for cls_idx, t_start, t_end in segments[:5]:
        print(f"  Class {cls_idx}: [{t_start:.3f}, {t_end:.3f}]")


if __name__ == "__main__":
    main()
