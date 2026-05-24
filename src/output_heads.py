"""Output Heads: Activity Classification, Temporal Localization, Scene Graph.

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

    Attention-pooling → 2-layer MLP → logits.
    Dropout=0.3: enough to regularise on small datasets without collapsing gradients.
    The wider 256→512 expansion was tried with dropout=0.4 but caused act_loss to
    collapse to 0.011 and mAP to regress — reverted to the simpler design.
    """

    def __init__(self, d_model=256, num_classes=157, dropout=0.3):
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

    For each ordered pair of K=5 objects, concatenates pair features + optional
    spatial bbox offset (Δx, Δy, Δw, Δh) → MLP → relation class.

    The 4 spatial features help the model learn spatial predicates such as
    'in front of', 'behind', 'leaning on' that depend on relative position.
    """

    def __init__(self, d_model=256, num_relations=11, dropout=0.3):
        super().__init__()
        # Input: concat of subject + object features (d_model*2)
        #        + 4 spatial offsets (Δx, Δy, Δw, Δh) when bbox_coords provided
        self._pair_in = d_model * 2 + 4
        self.relation_mlp = nn.Sequential(
            nn.Linear(self._pair_in, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_relations),
        )
        self.k = K

    def forward(self, fused, object_tokens, bbox_coords=None):
        """
        fused:         [B, 32, 256]    — spatially grounded temporal context
        object_tokens: [B, 32, 5, 256] — object tokens from encoder output
        bbox_coords:   [B, 32, 5, 4]  — (cx, cy, w, h) normalised to [0,1]
                       optional; if None, spatial offset is zero-filled.

        Returns: [B, 32, 20, num_relations] — relation logits per ordered pair
        """
        B, T, K_actual, D = object_tokens.shape

        # Enrich object tokens with temporal context
        temporal_ctx = fused.unsqueeze(2).expand(B, T, K_actual, D)  # [B, 32, 5, 256]
        enriched = object_tokens + temporal_ctx  # [B, 32, 5, 256]

        # Normalise bbox_coords or use zeros
        if bbox_coords is not None:
            # bbox_coords: [B, 32, 5, 4]
            bboxes = bbox_coords
        else:
            bboxes = torch.zeros(B, T, K_actual, 4, device=fused.device, dtype=fused.dtype)

        # Build all ordered pairs for each frame
        pairs = []
        for i in range(K_actual):
            for j in range(K_actual):
                if i == j:
                    continue
                subj = enriched[:, :, i, :]            # [B, 32, 256]
                obj  = enriched[:, :, j, :]            # [B, 32, 256]
                # Spatial offset: bbox_j - bbox_i  [B, 32, 4]
                spatial = bboxes[:, :, j, :] - bboxes[:, :, i, :]
                pair_feat = torch.cat([subj, obj, spatial], dim=-1)  # [B, 32, 260]
                pairs.append(pair_feat)

        # Stack: [B, 32, 20, 260]
        pairs_tensor = torch.stack(pairs, dim=2)
        return self.relation_mlp(pairs_tensor)  # [B, 32, 20, num_relations]


# ── Loss functions ──────────────────────────────────────────────────────────


def activity_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Focal Loss for multi-label activity classification.

    Focal loss down-weights easy negatives, helping the model focus on hard
    positives across 157 imbalanced classes.

        FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        logits:  [B, num_classes]
        targets: [B, num_classes] — multi-hot binary labels
        alpha:   balancing factor (default 0.25)
        gamma:   focusing exponent  (default 2.0)
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = torch.exp(-bce)                          # probability of correct class
    focal_weight = alpha * (1.0 - p_t) ** gamma
    return (focal_weight * bce).mean()


def giou_localization_loss(predictions, targets, activity_mask):
    """1-D Generalized IoU loss for temporal segment regression.

    Directly penalises span inflation — a model predicting [0.0, 1.0] for a
    true window of [0.3, 0.6] gets a large GIoU penalty, unlike L1 which
    rewards partial overlap regardless of span size.

    GIoU_1D = IoU - (enclosing_span - union) / enclosing_span
             ∈ [-1, 1];  loss = 1 - GIoU  ∈ [0, 2]

    Args:
        predictions:   [B, 32, 2]  (start, end) in [0, 1]
        targets:       [B, 32, 2]  (start, end) in [0, 1]
        activity_mask: [B, 32]     True where frame is within an activity
    """
    if activity_mask.sum() == 0:
        return predictions.new_tensor(0.0)

    pred   = predictions[activity_mask]   # [N, 2]
    target = targets[activity_mask]       # [N, 2]

    p_s, p_e = pred[:, 0],   pred[:, 1]    # predicted start / end
    t_s, t_e = target[:, 0], target[:, 1]  # target   start / end

    # Clamp so start <= end (Sigmoid guarantees [0,1] but not ordering)
    p_s, p_e = torch.min(p_s, p_e), torch.max(p_s, p_e)
    t_s, t_e = torch.min(t_s, t_e), torch.max(t_s, t_e)

    # Intersection
    inter = (torch.min(p_e, t_e) - torch.max(p_s, t_s)).clamp(min=0)  # [N]

    # Union
    p_len   = (p_e - p_s).clamp(min=0)
    t_len   = (t_e - t_s).clamp(min=0)
    union   = p_len + t_len - inter  # [N]

    iou = inter / (union + 1e-6)  # [N]

    # Enclosing span (smallest interval containing both)
    enclosing = (torch.max(p_e, t_e) - torch.min(p_s, t_s)).clamp(min=1e-6)  # [N]

    giou = iou - (enclosing - union) / enclosing  # [N] ∈ [-1, 1]
    return (1.0 - giou).mean()  # ∈ [0, 2]


# Keep the old L1 version accessible for ablation
def localization_loss(predictions, targets, activity_mask):
    """L1 loss over frames within annotated activity windows (legacy)."""
    if activity_mask.sum() == 0:
        return predictions.new_tensor(0.0)
    masked_pred   = predictions[activity_mask]
    masked_target = targets[activity_mask]
    return F.l1_loss(masked_pred, masked_target)


def scene_graph_loss(logits, targets):
    """Weighted cross-entropy for relation classification.

    Uses inverse-frequency class weights derived from the actual label
    distribution so rare predicates receive proportionally larger gradients.

    Relation index → label count (from data/labels.json diagnostic):
        0  background  : implicit / never stored  →  weight 0.0 (all bg pairs are -1)
        1  sitting on  : 29,511                   →  weight 2.53
        2  standing on :    417                   →  weight ×high (rare)
        3  holding     :      0                   →  weight 0.0 (absent from data)
        4  looking at  : 33,176                   →  weight 2.24
        5  eating      :      0                   →  weight 0.0 (absent from data)
        6  next to     :      0                   →  weight 0.0 (absent from data)
        7  in front of : 74,465  (most common)    →  weight 1.00 (reference)
        8  behind      : 31,832                   →  weight 2.34
        9  lying on    :    307                   →  weight ×high (rare)
       10  leaning on  : 10,178                   →  weight 7.32

    Classes with 0 instances get weight 0 — forcing the model to ignore them
    entirely rather than waste capacity trying to learn impossible classes.

    Args:
        logits:  [B, 32, 20, 11]
        targets: [B, 32, 20] — relation class indices (-1 = ignore)
    """
    B, T, P, C = logits.shape
    flat_targets = targets.view(B * T * P)

    # Guard against all-ignore targets (produces NaN)
    if (flat_targets != -1).sum() == 0:
        return logits.new_tensor(0.0)

    # Inverse-frequency weights — computed from full dataset label counts.
    # Reference = most-common class (in_front_of, 74 465 instances) → 1.0.
    # Classes absent from the dataset get weight 0.0.
    REF = 74_465.0
    _counts = torch.tensor([
        0.0,        # 0  background  — all bg pairs stored as -1 (ignored)
        REF / 29_511,  # 1  sitting on  → 2.53
        REF /    417,  # 2  standing on → 178.6  (capped below)
        0.0,        # 3  holding     — zero instances in data
        REF / 33_176,  # 4  looking at  → 2.24
        0.0,        # 5  eating      — zero instances in data
        0.0,        # 6  next to     — zero instances in data
        1.0,        # 7  in front of → 1.00  (reference)
        REF / 31_832,  # 8  behind      → 2.34
        REF /    307,  # 9  lying on    → 242.6 (capped below)
        REF / 10_178,  # 10 leaning on  → 7.32
    ], dtype=torch.float)

    # Cap extreme weights. Reverted to 12 (was briefly 25) — the higher cap
    # caused sg_loss to triple on small datasets, destabilising training.
    _counts = _counts.clamp(max=12.0)

    return F.cross_entropy(
        logits.view(B * T * P, C),
        flat_targets,
        weight=_counts.to(logits.device),
        ignore_index=-1,
    )


def total_loss(act_logits, loc_preds, sg_logits,
               act_targets, loc_targets, act_mask, sg_targets):
    """Weighted multi-task loss.

    L_total = L_activity + 0.3 × L_localization_giou + 0.3 × L_scene_graph

    Changes vs original:
    - localization: L1  →  GIoU  (prevents span-inflation collapse)
    - localization weight: 0.5  →  0.3  (stop SG/activity being crowded out)
    """
    L_act = activity_loss(act_logits, act_targets)
    L_loc = giou_localization_loss(loc_preds, loc_targets, act_mask)
    L_sg  = scene_graph_loss(sg_logits, sg_targets)
    return L_act + 0.3 * L_loc + 0.3 * L_sg, {
        "activity": L_act.item(),
        "localization": L_loc.item(),
        "scene_graph": L_sg.item(),
    }


# ── Inference decoding ─────────────────────────────────────────────────────


def decode_segments(
    predictions,
    activity_probs,
    threshold=0.15,
    video_duration=None,
    min_seg_ratio=0.10,
    max_seg_ratio=0.90,
    top_k_fallback=5,
):
    """Decode temporal segments from per-frame predictions.

    Args:
        predictions:    [T, 2]  — (start, end) per frame, normalized [0,1]
        activity_probs: [num_classes] — per-class probabilities (sigmoid)
        threshold:      minimum probability to consider a class active.
                        Lowered to 0.15 to match typical sigmoid output range.
        video_duration: optional float — video duration in seconds
        min_seg_ratio:  minimum segment length as fraction of video (default 0.10)
        max_seg_ratio:  maximum segment length as fraction of video (default 0.90);
                        segments spanning >90% of the video are filtered as spurious.
        top_k_fallback: if no class exceeds `threshold`, take the top-k by prob.

    Returns:
        list of (class_idx, t_start, t_end)
        If video_duration provided, t_start/t_end are in seconds; else normalized.
    """
    num_frames = predictions.shape[0]
    frame_positions = torch.linspace(0.0, 1.0, num_frames)  # [T]  # noqa: F841

    # ── 1. Decide which classes are active ────────────────────────────────────
    above_thresh = (activity_probs > threshold).nonzero(as_tuple=True)[0]
    if len(above_thresh) == 0:
        # Fallback: always return top-k predictions so output is never empty
        topk_vals, topk_idx = activity_probs.topk(top_k_fallback)
        above_thresh = topk_idx

    # Sort by probability descending so we emit highest-confidence first
    probs_sorted = above_thresh[activity_probs[above_thresh].argsort(descending=True)]

    segments = []
    seen_intervals = []  # for deduplication

    for cls_idx in probs_sorted:
        cls_idx_int = cls_idx.item()
        prob = activity_probs[cls_idx].item()

        t_starts = predictions[:, 0]  # [T]
        t_ends   = predictions[:, 1]  # [T]

        # ── 2. Use per-frame centre point to identify the active temporal window
        # The localization head predicts (start, end) for each frame position.
        # If the head is well-trained the predicted interval shrinks around the
        # true activity window.  We compute the midpoint each frame predicts and
        # cluster contiguous frames that agree within ±0.5 frame step.
        mids = ((t_starts + t_ends) / 2).clamp(0, 1)  # [T] mid-points

        # Compute frame-level confidence: shorter predicted span = more precise
        durations = (t_ends - t_starts).clamp(min=1e-3)  # [T]
        confidence = 1.0 / (durations + 1e-3)            # [T] inverse duration
        confidence = confidence / confidence.sum()         # normalise

        # Weighted median of start / end across all frames
        order = t_starts.argsort()
        cum_w = confidence[order].cumsum(0)
        median_idx = (cum_w >= 0.5).nonzero(as_tuple=True)[0]
        if len(median_idx) == 0:
            median_idx_val = 0
        else:
            median_idx_val = median_idx[0].item()
        t_start = t_starts[order[median_idx_val]].item()

        order_e = t_ends.argsort()
        cum_w_e = confidence[order_e].cumsum(0)
        median_idx_e = (cum_w_e >= 0.5).nonzero(as_tuple=True)[0]
        if len(median_idx_e) == 0:
            median_idx_e_val = 0
        else:
            median_idx_e_val = median_idx_e[0].item()
        t_end = t_ends[order_e[median_idx_e_val]].item()

        # ── 3. Sanity / degenerate-span correction ────────────────────────────
        t_start = float(max(0.0, min(t_start, 1.0)))
        t_end   = float(max(0.0, min(t_end,   1.0)))

        if t_end <= t_start:
            t_end = min(t_start + min_seg_ratio, 1.0)

        seg_len = t_end - t_start

        # Filter segments that are too short (< 10% of video) after correction
        if seg_len < min_seg_ratio:
            t_end = min(t_start + min_seg_ratio, 1.0)
            seg_len = t_end - t_start

        # Filter segments that span almost the entire video (localization collapsed)
        # — these are the degenerate [0.008, 0.991] outputs.
        if seg_len > max_seg_ratio:
            # Fall back to frame-position-based heuristic:
            # pick the temporal third where the most frames place their midpoint
            thirds = (mids * 3).long().clamp(0, 2)  # 0, 1, 2
            best_third = thirds.bincount(minlength=3).argmax().item()
            t_start = round(best_third / 3.0, 3)
            t_end   = round(min((best_third + 1) / 3.0, 1.0), 3)

        # ── 4. Deduplication — skip if very similar interval already added ────
        duplicate = False
        for prev_s, prev_e in seen_intervals:
            overlap = min(t_end, prev_e) - max(t_start, prev_s)
            union   = max(t_end, prev_e) - min(t_start, prev_s)
            if union > 0 and (overlap / union) > 0.7:
                duplicate = True
                break
        if duplicate:
            continue

        seen_intervals.append((t_start, t_end))

        # ── 5. Convert to seconds if duration is known ────────────────────────
        if video_duration is not None:
            t_start_out = round(t_start * video_duration, 1)
            t_end_out   = round(t_end   * video_duration, 1)
        else:
            t_start_out = round(t_start, 3)
            t_end_out   = round(t_end,   3)

        segments.append((cls_idx_int, t_start_out, t_end_out))

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
