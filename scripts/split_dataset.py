#!/usr/bin/env python3
"""Step 6 — Stratified dataset split (70/15/15).

Reference: doc 01 Step 6.
- Split by video ID, stratified by activity class distribution
- Fixed split (not regenerated between experiments)
"""
import json
import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split


def get_stratification_key(label_entry):
    """Get a single stratification key from multi-label activities.

    Uses the first active activity class for stratification (sklearn
    requires single-label). This approximates multi-label distribution.
    """
    activities = label_entry.get("activity", [])
    if activities:
        return activities[0]
    return -1  # no activity


def main():
    parser = argparse.ArgumentParser(description="Stratified 70/15/15 split")
    parser.add_argument("--labels", type=str, default="data/labels.json")
    parser.add_argument("--out", type=str, default="data/split.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.labels, "r") as f:
        labels = json.load(f)

    video_ids = sorted(labels.keys())
    strat_keys = [get_stratification_key(labels[vid]) for vid in video_ids]

    # First split: 70% train, 30% temp
    train_ids, temp_ids, train_keys, temp_keys = train_test_split(
        video_ids, strat_keys,
        test_size=0.30,
        random_state=args.seed,
        stratify=strat_keys,
    )

    # Second split: 50/50 of the 30% → 15% val, 15% test
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.50,
        random_state=args.seed,
        stratify=temp_keys,
    )

    split = {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(split, f, indent=2)

    print(f"Split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
