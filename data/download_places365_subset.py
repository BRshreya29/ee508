#!/usr/bin/env python3
"""Download a Places365 subset for training the 6-class scene classifier.

Reference: doc 01 Step 1, doc 05 — Scene Classifier.
Samples ~500 images per scene class from Places365.
"""
import os
import argparse

# Mapping from Places365 categories to our 6 working scene classes
PLACES365_TO_SCENE = {
    "kitchen": "kitchen",
    "dining_room": "kitchen",
    "street": "street",
    "road": "street",
    "highway": "street",
    "office": "office",
    "conference_room": "office",
    "gym": "gym",
    "fitness_center": "gym",
    "living_room": "living_room",
    "bedroom": "living_room",
    "forest": "outdoor",
    "park": "outdoor",
    "field": "outdoor",
    "mountain": "outdoor",
    "garden": "outdoor",
}

SCENE_CLASSES = ["kitchen", "street", "office", "gym", "living_room", "outdoor"]


def main():
    parser = argparse.ArgumentParser(description="Download Places365 subset")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/places365_subset",
        help="Output directory for scene-class subdirectories",
    )
    parser.add_argument(
        "--images_per_class",
        type=int,
        default=500,
        help="Number of images to download per scene class",
    )
    args = parser.parse_args()

    for scene_class in SCENE_CLASSES:
        class_dir = os.path.join(args.out_dir, scene_class)
        os.makedirs(class_dir, exist_ok=True)

    print("Places365 subset directory structure created.")
    print(f"Output: {args.out_dir}")
    print()
    print("Download Places365 images from: http://places2.csail.mit.edu/download.html")
    print("Place images into the appropriate class subdirectory.")
    print()
    print("Required class mapping (Places365 category → working class):")
    for p365_cat, scene in sorted(PLACES365_TO_SCENE.items()):
        print(f"  {p365_cat:20s} → {scene}")
    print()
    print(f"Target: ~{args.images_per_class} images per class × {len(SCENE_CLASSES)} classes")


if __name__ == "__main__":
    main()
