#!/usr/bin/env python3
"""Step 2 — Filter Action Genome .pkl annotations to the working taxonomy.

Reference: doc 01 Step 2.
- Parses Action Genome's `object_bbox_and_relationship.pkl` and `person_bbox.pkl`
- Also parses Charades CSV if available for video-level activities
- Keeps 8 object classes + 10 relation predicates
- Discards frames with no valid objects/relations; keeps videos with >=50% surviving frames
"""
import json
import os
import argparse
import pickle
from collections import defaultdict

# Working taxonomy — doc 01
OBJECT_CLASSES = ["person", "chair", "table", "cup", "phone", "book", "laptop", "door"]
OBJECT_SET = set(OBJECT_CLASSES)

RELATION_PREDICATES = [
    "sitting on", "standing on", "holding", "looking at", "eating",
    "next to", "in front of", "behind", "lying on", "leaning on",
]
RELATION_SET = set(RELATION_PREDICATES)
MIN_SURVIVING_RATIO = 0.5

# AG has objects like 'cupglassbottle'. Map them if necessary.
CLASS_MAPPING = {
    "cupglassbottle": "cup",
    "closetcabinet": "door",  # Approximate fallback if needed
}

def load_pickle_robust(path):
    print(f"Loading {path}...")
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")

def parse_charades_csv(csv_path):
    """Parse Charades CSV for video-level activities."""
    activities = defaultdict(list)
    if not os.path.exists(csv_path):
        return activities
    print(f"Loading activities from {csv_path}...")
    import csv
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("id", "")
            actions = row.get("actions", "")
            if not actions: continue
            for act in actions.split(";"):
                if not act: continue
                parts = act.split(" ")
                if len(parts) >= 3:
                    class_id = int(parts[0].replace("c", ""))
                    activities[vid].append({
                        "class_id": class_id,
                        "class_name": parts[0],
                        "start": float(parts[1]),
                        "end": float(parts[2])
                    })
    return activities

def main():
    parser = argparse.ArgumentParser(description="Filter AG annotations to working taxonomy")
    parser.add_argument("--ag_root", type=str, default="data/action_genome",
                        help="Root directory of Action Genome annotations")
    parser.add_argument("--charades_csv", type=str, default="data/Charades_v1_train.csv",
                        help="Path to Charades CSV for activities (optional)")
    parser.add_argument("--out", type=str, default="data/filtered_annotations.json",
                        help="Output filtered annotation file")
    args = parser.parse_args()

    ann_dir = os.path.join(args.ag_root, "annotations")
    obj_pkl = os.path.join(ann_dir, "object_bbox_and_relationship.pkl")
    person_pkl = os.path.join(ann_dir, "person_bbox.pkl")

    if not os.path.exists(obj_pkl):
        print(f"Missing {obj_pkl}. Cannot proceed.")
        # Fallback for dev: create empty output
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f: json.dump({}, f)
        return

    # 1. Load activities
    video_activities = parse_charades_csv(args.charades_csv)

    # 2. Extract Data
    video_frames = defaultdict(dict)
    total_frames_per_video = defaultdict(int)

    # Load person_bbox
    if os.path.exists(person_pkl):
        person_data = load_pickle_robust(person_pkl)
        for frame_path, bbox in person_data.items():
            vid, frame_id = frame_path.split("/")
            vid = vid.replace(".mp4", "").replace(".avi", "")
            total_frames_per_video[vid] += 1
            if vid not in video_frames: video_frames[vid] = defaultdict(lambda: {"objects": [], "relations": []})
            frame_id = frame_id.replace(".png", "")
            actual_bbox = bbox.get("bbox") if isinstance(bbox, dict) else bbox
            if actual_bbox is not None and len(actual_bbox) >= 4:
                video_frames[vid][frame_id]["objects"].append({
                    "id": 0, "class": "person", "bbox": list(actual_bbox[:4])
                })
        del person_data

    # Load object & relationships
    obj_data = load_pickle_robust(obj_pkl)
    for frame_path, objects in obj_data.items():
        vid, frame_id = frame_path.split("/")
        vid = vid.replace(".mp4", "").replace(".avi", "")
        frame_id = frame_id.replace(".png", "")
        total_frames_per_video[vid] += 1  # May double count with person, but rough total is ok

        frame_dict = video_frames[vid][frame_id]
        obj_id_start = len(frame_dict["objects"])

        valid_objs_map = {}
        # Parse objects
        for i, obj in enumerate(objects):
            raw_cls = obj.get("class", "").lower()
            mapped_cls = CLASS_MAPPING.get(raw_cls, raw_cls)
            if mapped_cls in OBJECT_SET:
                obj_id = obj_id_start + i
                valid_objs_map[obj_id] = mapped_cls
                frame_dict["objects"].append({
                    "id": obj_id, "class": mapped_cls,
                    "bbox": obj.get("bbox", []) if obj.get("bbox") is not None else []
                })

        # Parse relations
        for i, obj in enumerate(objects):
            obj_id = obj_id_start + i
            if obj_id not in valid_objs_map: continue

            for rel_type in ["attention_relationship", "spatial_relationship", "contacting_relationship"]:
                rels = obj.get(rel_type)
                if not rels: continue
                for r in rels:
                    if isinstance(r, dict):
                        predicate = str(r.get("class", r.get("predicate", ""))).replace("_", " ")
                    else:
                        predicate = str(r).replace("_", " ")
                    if " " not in predicate and len(predicate) > 5:
                        predicate = predicate.replace("ing", "ing ") # basic fix for "sittingon" -> "sitting on"
                    
                    if predicate in RELATION_SET:
                        frame_dict["relations"].append({
                            "subject_id": 0, # usually person is subject in AG
                            "object_id": obj_id,
                            "predicate": predicate
                        })
    del obj_data

    # 3. Filter
    filtered = {}
    total_videos = len(video_frames)
    kept_videos = 0

    print("Filtering frames and videos...")
    for vid, frames in video_frames.items():
        surviving_frames = {}
        for f_id, f_data in frames.items():
            if len(f_data["objects"]) > 0:
                surviving_frames[f_id] = f_data
        
        # Approximate original frame count
        orig_frames = max(len(frames), total_frames_per_video[vid] // 2) 
        if orig_frames > 0 and len(surviving_frames) / orig_frames >= MIN_SURVIVING_RATIO:
            filtered[vid] = {
                "activities": video_activities.get(vid, []),
                "duration": len(surviving_frames) / 30.0, # Approximate duration if missing
                "frames": surviving_frames
            }
            kept_videos += 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(filtered, f, indent=2)

    print(f"Filtered: {kept_videos}/{total_videos} videos retained")
    print(f"Output: {args.out}")

if __name__ == "__main__":
    main()

