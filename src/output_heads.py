"""Stage C — Output Heads (Activity, Localization, Scene Graph).

Reference: doc 09 — Output Heads.
All three heads branch from the shared fused representation [B, 32, 256].
Joint loss: L_total = L_activity + 0.5 * L_localization + 0.3 * L_scene_graph
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants
K = 5             # object tokens per frame
NUM_PAIRS = K * (K - 1)  # 20 ordered pairs

RELATION_CLASSES = [
    "background",    # class 0 — no relation
    "sitting on",
    "standing on",
    "holding",
    "looking at",
    "eating",
    "next to",
    "in front of",
    "behind",
    "lying on",
    "leaning on",
]


class ActivityClassificationHead(nn.Module):
    """Head C1 — Multi-label activity classification with attention pooling.

    Attention-pooling → 2-layer MLP → logits (applied sigmoid at loss time).
    """

    def __init__(self, d_model=256, num_classes=157, dropout=0.2):
        super().__init__()
        self.attn_query = nn.Parameter(torch.randn(d_model))
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, fused):
        """
        fused: [B, 32, 256]
        Returns: [B, num_classes] — raw logits (apply sigmoid at loss time)
        """
        # Attention pooling
        scores = torch.einsum("btd,d->bt", fused, self.attn_query)  # [B, 32]
        weights = scores.softmax(dim=-1).unsqueeze(-1)               # [B, 32, 1]
        pooled = (fused * weights).sum(dim=1)                        # [B, 256]
        return self.mlp(pooled)  # [B, num_classes]


class LocalizationHead(nn.Module):
    """Head C2 — Per-token temporal localization regression.

    Per-token MLP → (t_start, t_end) ∈ [0, 1].
    """

    def __init__(self, d_model=256, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
            nn.Sigmoid(),  # constrain to [0, 1]
        )

    def forward(self, fused):
        """
        fused: [B, 32, 256]
        Returns: [B, 32, 2] — (start, end) per frame position
        """
        return self.mlp(fused)


class SceneGraphHead(nn.Module):
    """Head C3 — Pairwise object relation classification.

    For each ordered pair of K=5 objects, concatenate pair features → MLP → relation class.
    """

    def __init__(self, d_model=256, num_relations=11, dropout=0.1):
        super().__init__()
        # num_relations = 10 predicates + 1 background (class 0)
        self.relation_mlp = nn.Sequential(
            nn.Linear(d_model * 2, 128),  # concat of subject + object features
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_relations),
        )
        self.k = K

    def forward(self, fused, object_tokens):
        """
        fused:         [B, 32, 256]    — spatially grounded temporal context
        object_tokens: [B, 32, 5, 256] — object tokens from encoder output

        Returns: [B, 32, 20, num_relations] — relation logits per ordered pair
        """
        B, T, K_actual, D = object_tokens.shape

        # Enrich object tokens with temporal context
        temporal_ctx = fused.unsqueeze(2).expand(B, T, K_actual, D)  # [B, 32, 5, 256]
        enriched = object_tokens + temporal_ctx  # [B, 32, 5, 256]

        # Build all ordered pairs for each frame
        pairs = []
        for i in range(K_actual):
            for j in range(K_actual):
                if i == j:
                    continue
                subj = enriched[:, :, i, :]  # [B, 32, 256]
                obj = enriched[:, :, j, :]   # [B, 32, 256]
                pair_feat = torch.cat([subj, obj], dim=-1)  # [B, 32, 512]
                pairs.append(pair_feat)

        # Stack: [B, 32, 20, 512]
        pairs_tensor = torch.stack(pairs, dim=2)
        return self.relation_mlp(pairs_tensor)  # [B, 32, 20, num_relations]


# ── Loss functions ──────────────────────────────────────────────────────────


def activity_loss(logits, targets, smoothing=0.1):
    """BCE with label smoothing for multi-label activity classification.

    Args:
        logits:  [B, num_classes]
        targets: [B, num_classes] — multi-hot binary labels
    """
    targets_smooth = targets * (1 - smoothing) + 0.5 * smoothing
    return F.binary_cross_entropy_with_logits(logits, targets_smooth)


def localization_loss(predictions, targets, activity_mask):
    """L1 loss over frames within annotated activity windows.

    Args:
        predictions:   [B, 32, 2]
        targets:       [B, 32, 2]
        activity_mask: [B, 32] — True where frame is within an activity
    """
    if activity_mask.sum() == 0:
        return predictions.new_tensor(0.0)
    masked_pred = predictions[activity_mask]    # [N, 2]
    masked_target = targets[activity_mask]      # [N, 2]
    return F.l1_loss(masked_pred, masked_target)


def scene_graph_loss(logits, targets):
    """Cross-entropy for relation classification.

    Args:
        logits:  [B, 32, 20, 11]
        targets: [B, 32, 20] — relation class indices (-1 = ignore)
    """
    B, T, P, C = logits.shape
    flat_targets = targets.view(B * T * P)

    # Guard against all-ignore targets (produces NaN)
    if (flat_targets != -1).sum() == 0:
        return logits.new_tensor(0.0)

    return F.cross_entropy(
        logits.view(B * T * P, C),
        flat_targets,
        ignore_index=-1,
    )


def total_loss(act_logits, loc_preds, sg_logits,
               act_targets, loc_targets, act_mask, sg_targets):
    """Weighted multi-task loss.

    L_total = L_activity + 0.5 × L_localization + 0.3 × L_scene_graph
    """
    L_act = activity_loss(act_logits, act_targets)
    L_loc = localization_loss(loc_preds, loc_targets, act_mask)
    L_sg = scene_graph_loss(sg_logits, sg_targets)
    return L_act + 0.5 * L_loc + 0.3 * L_sg, {
        "activity": L_act.item(),
        "localization": L_loc.item(),
        "scene_graph": L_sg.item(),
    }


# ── Inference decoding ─────────────────────────────────────────────────────


def decode_segments(predictions, activity_probs, threshold=0.5):
    """Decode temporal segments from per-frame predictions.

    Args:
        predictions:    [32, 2]  — (start, end) per frame
        activity_probs: [num_classes] — per-class probabilities

    Returns:
        list of (class_idx, t_start, t_end)
    """
    segments = []
    active_classes = (activity_probs > threshold).nonzero(as_tuple=True)[0]
    for cls_idx in active_classes:
        t_start = predictions[:, 0].mean().item()
        t_end = predictions[:, 1].mean().item()
        segments.append((cls_idx.item(), t_start, t_end))
    return segments


def decode_triplets(logits, object_class_ids, threshold=0.5):
    """Decode scene graph triplets from relation logits.

    Args:
        logits:           [32, 20, 11]
        object_class_ids: [32, 5] — detected object class per slot

    Returns:
        list of (frame_idx, subject_class, predicate, object_class)
    """
    probs = logits.softmax(dim=-1)  # [32, 20, 11]
    triplets = []
    pair_idx = 0
    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            for t in range(32):
                best_rel = probs[t, pair_idx].argmax().item()
                if best_rel != 0 and probs[t, pair_idx, best_rel] > threshold:
                    triplets.append((
                        t,
                        object_class_ids[t, i].item(),
                        RELATION_CLASSES[best_rel],
                        object_class_ids[t, j].item(),
                    ))
            pair_idx += 1
    return triplets
