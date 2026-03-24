#!/usr/bin/env python3
"""Step 5 — Prepare labels for all three output heads.

Reference: doc 01 Step 5, doc 09 — Output Heads.
- Activity labels: binary multi-hot vector per video
- Temporal labels: normalized (start, end) per activity per frame
- Scene graph labels: per-frame (subject, predicate, object) triplet indices
"""
import json
import os
import argparse
import numpy as np

# Constants from docs
NUM_FRAMES = 32
OBJECT_CLASSES = ["person", "chair", "table", "cup", "phone", "book", "laptop", "door"]
OBJ_TO_IDX = {c: i for i, c in enumerate(OBJECT_CLASSES)}

RELATION_PREDICATES = [
    "background",  # class 0
    "sitting on", "standing on", "holding", "looking at", "eating",
    "next to", "in front of", "behind", "lying on", "leaning on",
]
REL_TO_IDX = {r: i for i, r in enumerate(RELATION_PREDICATES)}

K = 5  # object tokens per frame


def prepare_activity_labels(video_ann, num_classes=157):
    """Extract multi-hot activity label vector.

    Returns:
        activity_labels: list of int indices of active classes
    """
    activities = video_ann.get("activities", [])
    labels = []
    for act in activities:
        class_id = act.get("class_id", act.get("activity_id", -1))
        if 0 <= class_id < num_classes:
            labels.append(class_id)
    return sorted(set(labels))


def prepare_temporal_labels(video_ann, num_classes=157):
    """Extract normalized (start, end) per activity.

    Returns:
        dict mapping class_id → (start_norm, end_norm)
    """
    activities = video_ann.get("activities", [])
    duration = video_ann.get("duration", 1.0)
    if duration <= 0:
        duration = 1.0

    temporal = {}
    for act in activities:
        class_id = act.get("class_id", act.get("activity_id", -1))
        start = act.get("start", 0.0)
        end = act.get("end", duration)

        start_norm = max(0.0, min(1.0, start / duration))
        end_norm = max(0.0, min(1.0, end / duration))

        if class_id >= 0:
            temporal[class_id] = [start_norm, end_norm]

    return temporal


def prepare_scene_graph_labels(video_ann, vid, frame_indices_map):
    """Extract per-frame scene graph triplets and map to sequence indices (0-31)."""
    frames = video_ann.get("frames", {})
    sg_labels = {}
    
    indices = frame_indices_map.get(vid, [])
    if not indices:
        return sg_labels

    for frame_id_str, frame_ann in frames.items():
        try:
            raw_frame_id = int(frame_id_str)
        except ValueError:
            continue
            
        # Find closest sequence index 0-31
        closest_seq_idx = min(range(len(indices)), key=lambda i: abs(indices[i] - raw_frame_id))
        seq_idx_str = str(closest_seq_idx)
        
        if seq_idx_str not in sg_labels:
            sg_labels[seq_idx_str] = []

        relations = frame_ann.get("relations", [])
        for rel in relations:
            subj_id = rel.get("subject_id")
            obj_id = rel.get("object_id")
            predicate = rel.get("predicate", "").lower().strip()

            if (subj_id is not None and obj_id is not None
                    and predicate in REL_TO_IDX):
                sg_labels[seq_idx_str].append([
                    int(subj_id),
                    REL_TO_IDX[predicate],
                    int(obj_id),
                ])

    return sg_labels


def main():
    parser = argparse.ArgumentParser(description="Prepare labels for all 3 heads")
    parser.add_argument("--annotations", type=str, default="data/filtered_annotations.json")
    parser.add_argument("--out", type=str, default="data/labels.json")
    parser.add_argument("--num_activity_classes", type=int, default=157)
    args = parser.parse_args()

    with open(args.annotations, "r") as f:
        annotations = json.load(f)
        
    frame_indices_path = "data/frame_indices.json"
    frame_indices_map = {}
    if os.path.exists(frame_indices_path):
        with open(frame_indices_path, "r") as f:
            frame_indices_map = json.load(f)

    all_labels = {}
    for vid, video_ann in annotations.items():
        activity = prepare_activity_labels(video_ann, args.num_activity_classes)
        temporal = prepare_temporal_labels(video_ann, args.num_activity_classes)
        scene_graph = prepare_scene_graph_labels(video_ann, vid, frame_indices_map)

        all_labels[vid] = {
            "activity": activity,
            "temporal": temporal,
            "scene_graph": scene_graph,
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_labels, f, indent=2)

    print(f"Prepared labels for {len(all_labels)} videos")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
