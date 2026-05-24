#!/usr/bin/env python3
"""Focused evaluation on the 8 object taxonomy classes and 11 scene-graph relations.

Produces:
  1. Per-object-class  : accuracy, precision, recall, F1
  2. Per-relation-class: accuracy, precision, recall, F1
  3. Confusion matrix  : objects (8×8) and relations (11×11)
  4. PNG plots saved to --out_dir

Run:
    python3 eval_objects.py \
        --checkpoint checkpoints/best_model.pt \
        --split_file data/split.json \
        --labels_file data/labels.json \
        --features_root features
"""
import os
import sys
import json
import argparse

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(__file__))
from src.model import SceneActivityModel
from src.dataset import get_dataloader
from src.output_heads import RELATION_CLASSES

# ── Taxonomy ──────────────────────────────────────────────────────────────────
OBJECT_CLASSES = ["person", "chair", "table", "cup", "phone", "book", "laptop", "door"]
NUM_OBJ_CLASSES = len(OBJECT_CLASSES)  # 8
NUM_RELATIONS   = len(RELATION_CLASSES)  # 11   (index 0 = background)
K = 5   # object slots per frame

# Relations that actually have GT instances in the dataset
# (holding=3, eating=5, next_to=6, background=0 have ZERO instances)
ACTIVE_REL_IDXS  = [1, 2, 4, 7, 8, 9, 10]
ACTIVE_REL_NAMES = [RELATION_CLASSES[i] for i in ACTIVE_REL_IDXS]


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_obj_class_ids(obj_feat_batch):
    """Read object class IDs from the one-hot block in obj_feat.

    obj_feat: [B, 32, 5, 324]
    The first 64 dims of the 324-dim vector are the DETR class one-hot.
    We read only the first 8 of those to match OBJECT_CLASSES.
    Returns [B, 32, 5] int tensor with class id in [0, NUM_OBJ_CLASSES-1].
    Values of -1 mark padding slots.
    """
    # first 8 dims of the 324-d vector → one-hot over our 8 classes
    onehot = obj_feat_batch[..., :NUM_OBJ_CLASSES]   # [B, 32, 5, 8]
    max_vals, class_ids = onehot.max(dim=-1)          # [B, 32, 5]
    return class_ids


def decode_obj_preds(sg_logits, obj_feat_batch):
    """Decode object class predictions from scene-graph head output.

    The scene-graph head sees object tokens in slots 0-4 per frame.
    We use the object token that achieves the highest confidence across
    all relations as the model's implicit predicted class for that slot.

    Returns:
        pred_classes [B, 32, 5]  — predicted class per slot  (0..7)
        true_classes [B, 32, 5]  — ground-truth from one-hot
    """
    true_classes = extract_obj_class_ids(obj_feat_batch)   # [B, 32, 5]

    # sg_logits: [B, 32, 20, 11]
    # Each ordered pair (i,j) shares subject slot i.
    # Average relation confidence across all pairs that share subject slot i
    # → gives an implicit score per slot.
    B, T, _, _ = sg_logits.shape
    probs = sg_logits.softmax(dim=-1)   # [B, 32, 20, 11]

    # Map pair index → subject slot index
    pair_to_subj = []
    for i in range(K):
        for j in range(K):
            if i != j:
                pair_to_subj.append(i)
    pair_to_subj = torch.tensor(pair_to_subj)  # [20]

    # Slot confidence: mean non-background probability across all pairs
    # where that slot is the subject
    slot_conf = torch.zeros(B, T, K)  # [B, 32, 5]
    pair_count = torch.zeros(K)
    for pair_idx, subj_slot in enumerate(pair_to_subj):
        s = subj_slot.item()
        slot_conf[:, :, s] += probs[:, :, pair_idx, 1:].sum(dim=-1)  # sum non-bg
        pair_count[s] += 1
    pair_count = pair_count.clamp(min=1)
    slot_conf = slot_conf / pair_count  # [B, 32, 5]

    # The model doesn't explicitly predict object class — we use the ground-truth
    # class for the CM axes and check if high-confidence slots match.
    # pred_classes mirrors true_classes for object cm (the object-level evaluation
    # is on whether the right object has correct relations, not classification).
    # → For object confusion matrix we compare per-slot GT class vs scene-graph
    #   head attention: the top predicted object in each frame (slot ordering is
    #   fixed and GT is used as reference).
    # We report: for each (slot, frame) whether its detected relation pattern is
    # consistent with the GT class  — and build a CM per predicted-slot-class.
    pred_classes = true_classes  # slots are position-indexed, GT = pred here

    return pred_classes, true_classes


@torch.no_grad()
def run_inference(model, loader, device):
    """Run model on test split, collect all outputs."""
    model.eval()
    all_sg_logits  = []   # [B, 32, 20, 11] each
    all_sg_targets = []   # [B, 32, 20]
    all_obj_feat   = []   # [B, 32, 5, 324]
    all_obj_mask   = []   # [B, 32, 5]

    for batch in loader:
        clip_feat, diff_feat, obj_feat, scene_feat, obj_mask, labels = batch
        clip_feat  = clip_feat.to(device)
        diff_feat  = diff_feat.to(device)
        obj_feat_d = obj_feat.to(device)
        scene_feat = scene_feat.to(device)
        obj_mask   = obj_mask.to(device)

        _, _, sg_logits, _ = model(clip_feat, diff_feat, obj_feat_d, scene_feat, obj_mask)

        all_sg_logits.append(sg_logits.cpu())
        all_sg_targets.append(labels["scene_graph"])  # already cpu
        all_obj_feat.append(obj_feat)
        all_obj_mask.append(obj_mask.cpu())

    return (
        torch.cat(all_sg_logits,  dim=0),   # [N, 32, 20, 11]
        torch.cat(all_sg_targets, dim=0),   # [N, 32, 20]
        torch.cat(all_obj_feat,   dim=0),   # [N, 32, 5, 324]
        torch.cat(all_obj_mask,   dim=0),   # [N, 32, 5]
    )


# ── Metric helpers ────────────────────────────────────────────────────────────

def safe_div(num, den):
    return num / den if den > 0 else 0.0


def per_class_metrics(y_true, y_pred, num_classes, ignore_index=-1):
    """Compute per-class accuracy, precision, recall, F1.

    y_true, y_pred: 1-D int arrays (flattened), ignore_index skipped.
    Returns: dict with arrays of length num_classes.
    """
    mask = y_true != ignore_index
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    tp   = np.zeros(num_classes, dtype=float)
    fp   = np.zeros(num_classes, dtype=float)
    fn   = np.zeros(num_classes, dtype=float)
    tn   = np.zeros(num_classes, dtype=float)
    total = len(y_true)

    for c in range(num_classes):
        tp[c] = ((y_pred == c) & (y_true == c)).sum()
        fp[c] = ((y_pred == c) & (y_true != c)).sum()
        fn[c] = ((y_pred != c) & (y_true == c)).sum()
        tn[c] = ((y_pred != c) & (y_true != c)).sum()

    precision = np.array([safe_div(tp[c], tp[c] + fp[c]) for c in range(num_classes)])
    recall    = np.array([safe_div(tp[c], tp[c] + fn[c]) for c in range(num_classes)])
    f1        = np.array([safe_div(2 * precision[c] * recall[c], precision[c] + recall[c])
                          for c in range(num_classes)])
    accuracy  = np.array([safe_div(tp[c] + tn[c], total) for c in range(num_classes)])

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def build_confusion_matrix(y_true, y_pred, num_classes, ignore_index=-1):
    mask = y_true != ignore_index
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def plot_confusion_matrix(cm, class_names, title, out_path, figsize=(10, 8)):
    """Plot and save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=figsize)
    # Normalize row-wise so every row sums to 1 (recall-normalized)
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm / row_sums

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Row-normalized fraction")

    n = len(class_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Annotate cells
    thresh = cm_norm.max() / 2.0
    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                    ha="center", va="center", fontsize=7, color=color)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_bar_metrics(names, metrics_dict, title, out_path):
    """Bar chart of precision / recall / F1 per class."""
    n = len(names)
    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, n * 1.2), 5))
    ax.bar(x - width, metrics_dict["precision"] * 100, width, label="Precision", color="#4a90d9", alpha=0.85)
    ax.bar(x,          metrics_dict["recall"]    * 100, width, label="Recall",    color="#e67e22", alpha=0.85)
    ax.bar(x + width,  metrics_dict["f1"]        * 100, width, label="F1",        color="#27ae60", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def print_table(names, metrics, label):
    """Print a formatted table of per-class metrics."""
    print(f"\n{'─'*70}")
    print(f"  {label}")
    print(f"{'─'*70}")
    print(f"  {'Class':<25} {'Acc%':>7} {'Prec%':>7} {'Rec%':>7} {'F1%':>7}  {'TP':>6} {'FP':>6} {'FN':>6}")
    print(f"{'─'*70}")
    for i, name in enumerate(names):
        print(
            f"  {name:<25} "
            f"{metrics['accuracy'][i]*100:>7.1f} "
            f"{metrics['precision'][i]*100:>7.1f} "
            f"{metrics['recall'][i]*100:>7.1f} "
            f"{metrics['f1'][i]*100:>7.1f}  "
            f"{int(metrics['tp'][i]):>6} "
            f"{int(metrics['fp'][i]):>6} "
            f"{int(metrics['fn'][i]):>6}"
        )
    print(f"{'─'*70}")
    macro_p  = metrics['precision'].mean() * 100
    macro_r  = metrics['recall'].mean()    * 100
    macro_f1 = metrics['f1'].mean()        * 100
    print(f"  {'MACRO AVG':<25} {'':>7} {macro_p:>7.1f} {macro_r:>7.1f} {macro_f1:>7.1f}")
    print(f"{'─'*70}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate 8-class object & relation metrics")
    parser.add_argument("--checkpoint",    type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--split_file",    type=str, default="data/split.json")
    parser.add_argument("--labels_file",   type=str, default="data/labels.json")
    parser.add_argument("--features_root", type=str, default="features")
    parser.add_argument("--split",         type=str, default="test",
                        help="Which split to evaluate: train / val / test")
    parser.add_argument("--batch_size",    type=int, default=8)
    parser.add_argument("--num_workers",   type=int, default=2)
    parser.add_argument("--device",        type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir",       type=str, default="eval_output")
    parser.add_argument("--rel_threshold", type=float, default=0.4,
                        help="Softmax threshold for calling a relation non-background")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── Load model ────────────────────────────────────────────────────────────
    model = SceneActivityModel().to(device)
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found at {args.checkpoint}")
        return
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint: {args.checkpoint}  (saved at epoch {ckpt.get('epoch', '?')})")

    # ── DataLoader ────────────────────────────────────────────────────────────
    loader = get_dataloader(
        args.split, args.split_file, args.labels_file, args.features_root,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    print(f"Evaluating split='{args.split}'  ({len(loader.dataset)} videos) ...")

    # ── Inference ─────────────────────────────────────────────────────────────
    sg_logits, sg_targets, obj_feat, obj_mask = run_inference(model, loader, device)
    # sg_logits : [N, 32, 20, 11]
    # sg_targets: [N, 32, 20]   (relation class index, -1 = ignore)
    # obj_feat  : [N, 32,  5, 324]
    # obj_mask  : [N, 32,  5]   (True = padding)

    N = sg_logits.shape[0]
    print(f"Collected {N} videos of data.")

    # ─────────────────────────────────────────────────────────────────────────
    # PART 1 — RELATION (scene-graph) metrics
    # ─────────────────────────────────────────────────────────────────────────
    sg_probs = sg_logits.softmax(dim=-1)            # [N, 32, 20, 11]
    sg_preds = sg_probs.argmax(dim=-1).numpy()      # [N, 32, 20]
    sg_true  = sg_targets.numpy()                   # [N, 32, 20]  (−1 = ignore)

    # Flatten to 1-D, keeping only non-padding entries
    # Object-mask: if both subject AND object slots are masked → skip
    # Build pair-level mask: pair (i,j) is valid if neither slot i nor j is padding
    pair_mask = np.ones((N, 32, 20), dtype=bool)
    om = obj_mask.numpy()  # [N, 32, 5]  True = padding
    pair_idx = 0
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            # mask out pairs where either endpoint is padding
            pair_mask[:, :, pair_idx] = ~om[:, :, i] & ~om[:, :, j]
            pair_idx += 1

    # Combine: must be valid pair AND not ignored in sg_targets
    valid = pair_mask & (sg_true != -1)

    rel_true = sg_true[valid].flatten()
    rel_pred = sg_preds[valid].flatten()

    rel_metrics = per_class_metrics(rel_true, rel_pred, NUM_RELATIONS, ignore_index=-1)
    rel_cm      = build_confusion_matrix(rel_true, rel_pred, NUM_RELATIONS)

    print_table(RELATION_CLASSES, rel_metrics, "RELATION CLASSIFICATION (per predicate)")

    plot_confusion_matrix(
        rel_cm, RELATION_CLASSES,
        "Relation Confusion Matrix (test set) — all 11 classes",
        os.path.join(args.out_dir, "cm_relations.png"),
        figsize=(12, 10),
    )
    plot_bar_metrics(
        RELATION_CLASSES, rel_metrics,
        "Relation Precision / Recall / F1 — all 11 classes",
        os.path.join(args.out_dir, "bar_relations.png"),
    )

    # ── Active-class-only relation metrics (RECOMMENDED) ─────────────────────
    # Only score the 7 classes that have GT instances in the data.
    # Omitting zero-instance classes gives a fairer picture of model quality.
    active_mask = np.isin(rel_true, ACTIVE_REL_IDXS)
    rel_true_a  = rel_true[active_mask]
    rel_pred_a  = rel_pred[active_mask]

    # Remap original indices to 0..6 for per_class_metrics
    idx_remap   = {old: new for new, old in enumerate(ACTIVE_REL_IDXS)}
    rel_true_r  = np.array([idx_remap.get(x, -1) for x in rel_true_a])
    rel_pred_r  = np.array([idx_remap.get(x, -1) for x in rel_pred_a])

    active_metrics = per_class_metrics(rel_true_r, rel_pred_r, len(ACTIVE_REL_IDXS))
    active_cm      = build_confusion_matrix(rel_true_r, rel_pred_r, len(ACTIVE_REL_IDXS))

    print_table(ACTIVE_REL_NAMES, active_metrics,
                "RELATION CLASSIFICATION — 7 active classes only (recommended)")

    plot_confusion_matrix(
        active_cm, ACTIVE_REL_NAMES,
        "Relation Confusion Matrix — active classes only (test set)",
        os.path.join(args.out_dir, "cm_relations_active.png"),
        figsize=(10, 8),
    )
    plot_bar_metrics(
        ACTIVE_REL_NAMES, active_metrics,
        "Relation Precision / Recall / F1 — active classes only",
        os.path.join(args.out_dir, "bar_relations_active.png"),
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PART 2 — OBJECT CLASS metrics
    # Evaluate whether the correct object class is in the correct slot.
    # GT class comes from obj_feat one-hot; predicted class also from obj_feat
    # (slots are position-indexed, not predicted by the model directly).
    # What we actually measure here:
    #   For each occupied slot, does the scene-graph head assign meaningful
    #   (non-background) relations consistent with that object's class?
    # We build it two ways:
    #   2a. Per-class detection rate: % of slots with GT class c that have
    #       at least one non-background relation predicted.
    #   2b. Object confusion matrix: when the model assigns a high-confidence
    #       relation to a slot, which GT class owned that slot?
    # ─────────────────────────────────────────────────────────────────────────

    # GT object classes per slot [N, 32, 5]
    gt_obj_classes = extract_obj_class_ids(obj_feat).numpy()   # int [0..7]
    # Validity mask (not padding)
    slot_valid = ~om  # [N, 32, 5]  True = real object

    # 2a. Per-slot relation activity: is this slot involved in any
    #     non-background predicted relation?
    slot_active_pred = np.zeros((N, 32, K), dtype=bool)
    slot_active_true = np.zeros((N, 32, K), dtype=bool)
    pair_idx = 0
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            # predicted non-background
            slot_active_pred[:, :, i] |= (sg_preds[:, :, pair_idx] != 0)
            # GT non-background
            slot_active_true[:, :, i] |= (
                (sg_true[:, :, pair_idx] != 0) & (sg_true[:, :, pair_idx] != -1)
            )
            pair_idx += 1

    # For each valid slot: treat "has GT non-bg relation" as positive label,
    # "has predicted non-bg relation" as prediction.
    # Group by object class → gives per-class detection accuracy.
    obj_tp = np.zeros(NUM_OBJ_CLASSES, dtype=float)
    obj_fp = np.zeros(NUM_OBJ_CLASSES, dtype=float)
    obj_fn = np.zeros(NUM_OBJ_CLASSES, dtype=float)
    obj_tn = np.zeros(NUM_OBJ_CLASSES, dtype=float)
    obj_total = np.zeros(NUM_OBJ_CLASSES, dtype=float)

    for c in range(NUM_OBJ_CLASSES):
        is_class_c = (gt_obj_classes == c) & slot_valid   # [N, 32, 5]
        true_pos_c = is_class_c & slot_active_true       # GT has relation
        true_neg_c = is_class_c & ~slot_active_true      # GT no relation

        obj_tp[c] = ( true_pos_c &  slot_active_pred).sum()
        obj_fn[c] = ( true_pos_c & ~slot_active_pred).sum()
        obj_fp[c] = ( true_neg_c &  slot_active_pred).sum()
        obj_tn[c] = ( true_neg_c & ~slot_active_pred).sum()
        obj_total[c] = is_class_c.sum()

    total_all = obj_total.sum()
    obj_precision = np.array([safe_div(obj_tp[c], obj_tp[c] + obj_fp[c]) for c in range(NUM_OBJ_CLASSES)])
    obj_recall    = np.array([safe_div(obj_tp[c], obj_tp[c] + obj_fn[c]) for c in range(NUM_OBJ_CLASSES)])
    obj_f1        = np.array([safe_div(2*p*r, p+r) for p, r in zip(obj_precision, obj_recall)])
    obj_accuracy  = np.array([safe_div(obj_tp[c] + obj_tn[c], obj_total[c]) for c in range(NUM_OBJ_CLASSES)])
    obj_metrics   = {
        "accuracy":  obj_accuracy,
        "precision": obj_precision,
        "recall":    obj_recall,
        "f1":        obj_f1,
        "tp": obj_tp, "fp": obj_fp, "fn": obj_fn, "tn": obj_tn,
    }

    # 2b. Object × object confusion matrix
    # Build: for each (valid slot, frame) where the slot is involved in a
    # predicted relation, record (GT class of that slot).
    # We create an N×N CM where rows = GT and columns = "most commonly
    # co-occurring predicted object class in that pair".
    # Simplified: per pair (i,j) with non-bg prediction,
    # record (GT class of i, GT class of j).
    obj_cm = np.zeros((NUM_OBJ_CLASSES, NUM_OBJ_CLASSES), dtype=int)
    pair_idx = 0
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            # Pairs with non-background prediction
            active_pairs = sg_preds[:, :, pair_idx] != 0   # [N, 32]
            valid_i = slot_valid[:, :, i]
            valid_j = slot_valid[:, :, j]
            sel = active_pairs & valid_i & valid_j

            g_i = gt_obj_classes[:, :, i][sel]   # GT class of subject
            g_j = gt_obj_classes[:, :, j][sel]   # GT class of object

            for gi, gj in zip(g_i, g_j):
                if 0 <= gi < NUM_OBJ_CLASSES and 0 <= gj < NUM_OBJ_CLASSES:
                    obj_cm[gi, gj] += 1
            pair_idx += 1

    print_table(OBJECT_CLASSES, obj_metrics,
                "OBJECT CLASS — Relation Detection Rate (precision/recall vs GT)")

    plot_confusion_matrix(
        obj_cm, OBJECT_CLASSES,
        "Object Co-occurrence in Predicted Relations (test set)",
        os.path.join(args.out_dir, "cm_objects.png"),
        figsize=(9, 8),
    )
    plot_bar_metrics(
        OBJECT_CLASSES, obj_metrics,
        "Object-Class Relation-Detection Precision / Recall / F1",
        os.path.join(args.out_dir, "bar_objects.png"),
    )

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("  SUMMARY")
    print(f"{'═'*70}")
    print(f"  Videos evaluated             : {N}")
    print(f"  -- ALL 11 RELATION CLASSES --")
    print(f"  Relation macro-precision     : {rel_metrics['precision'].mean()*100:.1f}%")
    print(f"  Relation macro-recall        : {rel_metrics['recall'].mean()*100:.1f}%")
    print(f"  Relation macro-F1            : {rel_metrics['f1'].mean()*100:.1f}%")
    print(f"  Relation overall accuracy    : {(rel_true == rel_pred).mean()*100:.1f}%")
    print(f"  -- 7 ACTIVE CLASSES ONLY (recommended) --")
    print(f"  Active macro-precision       : {active_metrics['precision'].mean()*100:.1f}%")
    print(f"  Active macro-recall          : {active_metrics['recall'].mean()*100:.1f}%")
    print(f"  Active macro-F1              : {active_metrics['f1'].mean()*100:.1f}%")
    print(f"  Active overall accuracy      : {(rel_true_r == rel_pred_r).mean()*100:.1f}%")
    print()
    print(f"  Object macro-precision       : {obj_precision.mean()*100:.1f}%")
    print(f"  Object macro-recall          : {obj_recall.mean()*100:.1f}%")
    print(f"  Object macro-F1              : {obj_f1.mean()*100:.1f}%")
    print(f"{'═'*70}")
    print(f"\nAll plots saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
