#!/usr/bin/env python3
"""Training script for the SceneActivityModel.

Reference: doc 07 (training setup), doc 09 (joint loss).
- AdamW with per-group LR: projection 1e-4, transformer 2e-5, heads 5e-5 (default)
- Weight decay 1e-4, gradient clipping max_norm=1.0
- Linear warmup (5 epochs) + ReduceLROnPlateau watching mAP
- 80 epochs, logging per-task losses + mAP/mIoU/R@20
- Saves best_model.pt (min val_loss) AND best_map_model.pt (max mAP) separately
  so best_model.pt is NEVER overwritten by the mAP-best checkpoint.
"""
import os
import sys
import argparse
import time
import json
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.dirname(__file__))
from src.model import SceneActivityModel
from src.dataset import get_dataloader
from src.output_heads import total_loss


def freeze_backbone(model):
    """Freeze transformer + cross-attention so only heads are updated.

    Call after loading a checkpoint. The frozen modules still run forward
    passes (gradients are just detached), so no inference code changes.
    """
    frozen_modules = [model.transformer, model.cross_attention,
                      model.projection, model.pos_encoding]
    for module in frozen_modules:
        for p in module.parameters():
            p.requires_grad = False
    frozen_names = ["transformer", "cross_attention", "projection", "pos_encoding"]
    print(f"Frozen: {frozen_names}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters after freeze: {trainable:,}")


def build_optimizer(model, lr_proj=1e-4, lr_transformer=2e-5, lr_heads=5e-5, weight_decay=1e-3):
    """Build AdamW optimizer with per-group learning rates (doc 07).

    Only param groups with requires_grad=True params are included,
    so freeze_backbone() interacts cleanly with this function.
    """
    def _params(module):
        return [p for p in module.parameters() if p.requires_grad]

    param_groups = []
    if _params(model.projection):
        param_groups.append({"params": _params(model.projection),    "lr": lr_proj,        "name": "projection"})
    if _params(model.pos_encoding):
        param_groups.append({"params": _params(model.pos_encoding),  "lr": lr_proj,        "name": "pos_encoding"})
    if _params(model.transformer):
        param_groups.append({"params": _params(model.transformer),   "lr": lr_transformer, "name": "transformer"})
    if _params(model.cross_attention):
        param_groups.append({"params": _params(model.cross_attention), "lr": lr_proj,       "name": "cross_attn"})
    param_groups.append({"params": _params(model.activity_head),      "lr": lr_heads,       "name": "activity_head"})
    param_groups.append({"params": _params(model.localization_head),  "lr": lr_heads,       "name": "loc_head"})
    param_groups.append({"params": _params(model.scene_graph_head),   "lr": lr_heads,       "name": "sg_head"})
    # Drop empty groups (frozen modules)
    param_groups = [g for g in param_groups if len(g["params"]) > 0]
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def warmup_lr(optimizer, epoch, warmup_epochs=5):
    """Linear warmup: scale LR by epoch/warmup_epochs for the first N epochs."""
    if epoch < warmup_epochs:
        scale = (epoch + 1) / warmup_epochs
        for pg in optimizer.param_groups:
            pg["lr"] = pg.get("initial_lr", pg["lr"]) * scale


def train_one_epoch(model, dataloader, optimizer, device, epoch, warmup_epochs=5):
    """Train for one epoch."""
    model.train()
    total_samples = 0
    running_loss = 0.0
    loss_components = {"activity": 0.0, "localization": 0.0, "scene_graph": 0.0}

    warmup_lr(optimizer, epoch, warmup_epochs)

    for batch_idx, batch in enumerate(dataloader):
        clip_feat, diff_feat, obj_feat, scene_feat, obj_mask, labels = batch

        clip_feat = clip_feat.to(device)
        diff_feat = diff_feat.to(device)
        obj_feat = obj_feat.to(device)
        scene_feat = scene_feat.to(device)
        obj_mask = obj_mask.to(device)

        act_targets = labels["activity"].to(device)
        loc_targets = labels["temporal"].to(device)
        act_mask = labels["activity_mask"].to(device)
        sg_targets = labels["scene_graph"].to(device)

        # Forward
        act_logits, loc_preds, sg_logits, _ = model(
            clip_feat, diff_feat, obj_feat, scene_feat, obj_mask
        )

        # Loss
        loss, components = total_loss(
            act_logits, loc_preds, sg_logits,
            act_targets, loc_targets, act_mask, sg_targets
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = clip_feat.size(0)
        running_loss += loss.item() * bs
        total_samples += bs
        for k, v in components.items():
            loss_components[k] += v * bs

    avg_loss = running_loss / max(total_samples, 1)
    for k in loss_components:
        loss_components[k] /= max(total_samples, 1)

    return avg_loss, loss_components


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate on val/test set."""
    model.eval()
    total_samples = 0
    running_loss = 0.0
    loss_components = {"activity": 0.0, "localization": 0.0, "scene_graph": 0.0}

    all_act_logits = []
    all_act_targets = []

    for batch in dataloader:
        clip_feat, diff_feat, obj_feat, scene_feat, obj_mask, labels = batch

        clip_feat = clip_feat.to(device)
        diff_feat = diff_feat.to(device)
        obj_feat = obj_feat.to(device)
        scene_feat = scene_feat.to(device)
        obj_mask = obj_mask.to(device)

        act_targets = labels["activity"].to(device)
        loc_targets = labels["temporal"].to(device)
        act_mask = labels["activity_mask"].to(device)
        sg_targets = labels["scene_graph"].to(device)

        act_logits, loc_preds, sg_logits, _ = model(
            clip_feat, diff_feat, obj_feat, scene_feat, obj_mask
        )

        loss, components = total_loss(
            act_logits, loc_preds, sg_logits,
            act_targets, loc_targets, act_mask, sg_targets
        )

        bs = clip_feat.size(0)
        running_loss += loss.item() * bs
        total_samples += bs
        for k, v in components.items():
            loss_components[k] += v * bs

        all_act_logits.append(act_logits.cpu())
        all_act_targets.append(act_targets.cpu())

    avg_loss = running_loss / max(total_samples, 1)
    for k in loss_components:
        loss_components[k] /= max(total_samples, 1)

    # Compute mAP (simplified per-class AP)
    mAP = compute_mAP(torch.cat(all_act_logits), torch.cat(all_act_targets))

    return avg_loss, loss_components, mAP


def compute_mAP(logits, targets):
    """Compute mean Average Precision for multi-label classification."""
    from sklearn.metrics import average_precision_score
    import numpy as np

    probs = logits.sigmoid().numpy()
    targets_np = targets.numpy()

    # Only compute AP for classes that have at least one positive example
    aps = []
    for c in range(targets_np.shape[1]):
        if targets_np[:, c].sum() > 0:
            ap = average_precision_score(targets_np[:, c], probs[:, c])
            aps.append(ap)

    return float(np.mean(aps)) if aps else 0.0


def main():
    parser = argparse.ArgumentParser(description="Train SceneActivityModel")
    parser.add_argument("--split_file", type=str, default="data/split.json")
    parser.add_argument("--labels_file", type=str, default="data/labels.json")
    parser.add_argument("--features_root", type=str, default="features")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Max epochs; early stopping may terminate earlier.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze projection+transformer+cross_attention; only train the 3 output heads.")
    # LR / regularisation overrides
    parser.add_argument("--lr_proj", type=float, default=1e-4)
    parser.add_argument("--lr_transformer", type=float, default=2e-5)
    parser.add_argument("--lr_heads", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-3,
                        help="L2 weight decay (default 1e-3, 10× original 1e-4).")
    parser.add_argument("--early_stop_patience", type=int, default=12,
                        help="Stop training if mAP does not improve for this many epochs.")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # DataLoaders
    train_loader = get_dataloader(
        "train", args.split_file, args.labels_file, args.features_root,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    val_loader = get_dataloader(
        "val", args.split_file, args.labels_file, args.features_root,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    # Model
    model = SceneActivityModel().to(device)

    # Resume checkpoint BEFORE building optimizer so frozen state is respected
    start_epoch = 0
    best_val_loss = float("inf")
    best_mAP = 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_mAP = ckpt.get("mAP", 0.0)
        print(f"Resumed from epoch {start_epoch}")

    # Optionally freeze backbone AFTER loading weights
    if args.freeze_backbone:
        freeze_backbone(model)
    else:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {total_params:,}")

    # Optimizer — built after freeze so excluded params are skipped
    optimizer = build_optimizer(
        model,
        lr_proj=args.lr_proj,
        lr_transformer=args.lr_transformer,
        lr_heads=args.lr_heads,
        weight_decay=args.weight_decay,
    )
    # Store initial LR for warmup
    for pg in optimizer.param_groups:
        pg["initial_lr"] = pg["lr"]

    # ReduceLROnPlateau — decays LR when mAP stops improving.
    # patience=3: react faster on a small dataset where plateaus are short.
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
    )
    # Early-stopping state
    epochs_no_improve = 0

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()

        # Train
        train_loss, train_comp = train_one_epoch(
            model, train_loader, optimizer, device, epoch, args.warmup_epochs
        )

        # Evaluate
        val_loss, val_comp, mAP = evaluate(model, val_loader, device)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"[{elapsed:.1f}s] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"mAP={mAP:.4f} "
            f"act={val_comp['activity']:.4f} "
            f"loc={val_comp['localization']:.4f} "
            f"sg={val_comp['scene_graph']:.4f}"
        )

        # ── Checkpoint 1: best val_loss (original best_model.pt — NEVER renamed) ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "mAP": mAP,
            }, ckpt_path)
            print(f"  → Saved best_model.pt (val_loss={best_val_loss:.4f})")

        # ── Checkpoint 2: best mAP (separate file — never overwrites best_model.pt) ──
        if mAP > best_mAP:
            best_mAP = mAP
            epochs_no_improve = 0
            map_ckpt_path = os.path.join(args.checkpoint_dir, "best_map_model.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "mAP": best_mAP,
            }, map_ckpt_path)
            print(f"  → Saved best_map_model.pt (mAP={best_mAP:.4f})")
        else:
            epochs_no_improve += 1

        # ── Scheduler step: ReduceLROnPlateau watches mAP (skip during warmup) ──
        if epoch >= args.warmup_epochs:
            scheduler.step(mAP)

        # ── Early stopping ──
        if epochs_no_improve >= args.early_stop_patience:
            print(f"Early stopping: no mAP improvement for {args.early_stop_patience} epochs.")
            break

        # Save latest
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "mAP": mAP,
        }, os.path.join(args.checkpoint_dir, "latest_model.pt"))

    print("Training complete.")


if __name__ == "__main__":
    main()
