"""PyTorch Dataset and DataLoader for precomputed features.

Reads cached .pt feature files at training time. No video file is opened.
"""
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Constants
NUM_FRAMES = 32
K = 5  # objects per frame
NUM_ACTIVITY_CLASSES = 157
NUM_RELATIONS = 11  # 10 predicates + 1 background
NUM_PAIRS = K * (K - 1)  # 20 ordered pairs


class ActionGenomeFeatureDataset(Dataset):
    """Loads precomputed features from disk.

    Returns per-video:
        clip_feat:  [32, 512]
        diff_feat:  [32, 49]
        obj_feat:   [32, 5, 324]
        scene_feat: [32, 6]
        obj_mask:   [32, 5]   — True where object slot is padding
        labels:     dict with activity, temporal, scene_graph
    """

    def __init__(self, split_file, labels_file, features_root="features", split="train"):
        with open(split_file, "r") as f:
            splits = json.load(f)
        self.video_ids = splits[split]

        with open(labels_file, "r") as f:
            self.all_labels = json.load(f)

        self.features_root = features_root

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        feat_dir = os.path.join(self.features_root, vid)

        clip_feat = torch.load(os.path.join(feat_dir, "clip.pt"), weights_only=True)
        diff_feat = torch.load(os.path.join(feat_dir, "framediff.pt"), weights_only=True)
        obj_feat = torch.load(os.path.join(feat_dir, "objects.pt"), weights_only=True)
        scene_feat = torch.load(os.path.join(feat_dir, "scene.pt"), weights_only=True)

        # Object padding mask
        obj_mask_path = os.path.join(feat_dir, "obj_mask.pt")
        if os.path.exists(obj_mask_path):
            obj_mask = torch.load(obj_mask_path, weights_only=True)
        else:
            # Infer mask: all-zero tokens are padding
            obj_mask = (obj_feat.abs().sum(dim=-1) == 0)  # [32, 5]

        # Labels
        label_entry = self.all_labels.get(vid, {})
        labels = self._prepare_labels(label_entry)

        return clip_feat, diff_feat, obj_feat, scene_feat, obj_mask, labels

    def _prepare_labels(self, label_entry):
        """Convert raw label dict to tensors."""
        # Activity: multi-hot [num_classes]
        activity_indices = label_entry.get("activity", [])
        activity = torch.zeros(NUM_ACTIVITY_CLASSES)
        for idx in activity_indices:
            if 0 <= idx < NUM_ACTIVITY_CLASSES:
                activity[idx] = 1.0

        # Temporal: [32, 2] — per-frame (start, end)
        temporal = torch.full((NUM_FRAMES, 2), 0.5)  # default to midpoint
        activity_mask = torch.zeros(NUM_FRAMES, dtype=torch.bool)
        temporal_dict = label_entry.get("temporal", {})
        for cls_id_str, (start, end) in temporal_dict.items():
            # Mark frames within this activity's temporal extent
            start_frame = int(start * NUM_FRAMES)
            end_frame = int(end * NUM_FRAMES)
            for t in range(max(0, start_frame), min(NUM_FRAMES, end_frame + 1)):
                temporal[t] = torch.tensor([start, end])
                activity_mask[t] = True

        # Scene graph: [32, 20] — relation class per object pair per frame
        sg_targets = torch.full((NUM_FRAMES, NUM_PAIRS), -1, dtype=torch.long)
        sg_dict = label_entry.get("scene_graph", {})
        # Scene graph labels are per-frame triplets — simplified mapping
        for frame_id, triplets in sg_dict.items():
            frame_idx = int(frame_id) if frame_id.isdigit() else 0
            if frame_idx >= NUM_FRAMES:
                continue
            for triplet in triplets:
                if len(triplet) == 3:
                    subj_idx, pred_idx, obj_idx = triplet
                    # Map to pair index (ordered pairs, i != j)
                    pair_idx = self._get_pair_index(subj_idx, obj_idx)
                    if pair_idx >= 0:
                        sg_targets[frame_idx, pair_idx] = pred_idx

        return {
            "activity": activity,
            "temporal": temporal,
            "activity_mask": activity_mask,
            "scene_graph": sg_targets,
        }

    @staticmethod
    def _get_pair_index(subj_slot, obj_slot):
        """Map (subject_slot, object_slot) to ordered pair index (0-19)."""
        if subj_slot == obj_slot or subj_slot >= K or obj_slot >= K:
            return -1
        pair_idx = 0
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                if i == subj_slot and j == obj_slot:
                    return pair_idx
                pair_idx += 1
        return -1


def collate_fn(batch):
    """Custom collate for ActionGenomeFeatureDataset."""
    clip_feats, diff_feats, obj_feats, scene_feats, obj_masks, labels_list = zip(*batch)

    clip_batch = torch.stack(clip_feats)
    diff_batch = torch.stack(diff_feats)
    obj_batch = torch.stack(obj_feats)
    scene_batch = torch.stack(scene_feats)
    mask_batch = torch.stack(obj_masks)

    label_batch = {
        "activity": torch.stack([l["activity"] for l in labels_list]),
        "temporal": torch.stack([l["temporal"] for l in labels_list]),
        "activity_mask": torch.stack([l["activity_mask"] for l in labels_list]),
        "scene_graph": torch.stack([l["scene_graph"] for l in labels_list]),
    }

    return clip_batch, diff_batch, obj_batch, scene_batch, mask_batch, label_batch


def get_dataloader(split, split_file="data/split.json", labels_file="data/labels.json",
                   features_root="features", batch_size=16, num_workers=4, shuffle=None):
    """Factory function for DataLoader."""
    if shuffle is None:
        shuffle = (split == "train")

    dataset = ActionGenomeFeatureDataset(
        split_file=split_file,
        labels_file=labels_file,
        features_root=features_root,
        split=split,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == "train"),
    )
