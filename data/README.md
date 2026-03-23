# Data Directory Layout

```
data/
├── action_genome/            # Action Genome annotations (JSON per video)
│   ├── annotations/
│   └── ...
├── charades_videos/          # Raw Charades video files (.mp4)
├── places365_subset/         # ~500 images × 6 scene classes
│   ├── kitchen/
│   ├── street/
│   ├── office/
│   ├── gym/
│   ├── living_room/
│   └── outdoor/
├── filtered_annotations.json # Output of Step 2 — taxonomy-filtered AG annotations
├── labels.json               # Output of Step 5 — activity/temporal/SG labels
└── split.json                # Output of Step 6 — train/val/test video IDs

features/                     # Precomputed features (outside data/, at project root)
└── {video_id}/
    ├── clip.pt               # [32, 512]   — CLIP CLS tokens
    ├── framediff.pt          # [32, 49]    — frame-difference vectors
    ├── objects.pt            # [32, 5, 324] — DETR object tokens
    └── scene.pt              # [32, 6]     — scene classifier softmax

checkpoints/
└── scene_mlp.pt              # Trained scene classifier (offline)
```

## Disk Usage (estimated for ~9,848 videos)

| Component | Size |
|---|---|
| CLIP features | ~640 MB |
| Frame-diff features | ~290 MB |
| Object features | ~1.5 GB |
| Scene features | ~80 MB |
| Labels | ~50 MB |
| **Total features** | **~2.5 GB** |
