# EE508 — Long-Term Scene & Activity Understanding with Transformers

Scene context modeling and extended temporal reasoning over video using a shared Transformer encoder with three simultaneous output heads: activity classification, temporal localization, and scene graph generation.

---

## What This Project Does

Takes a video (5 seconds to 5 minutes), samples 32 frames uniformly, runs them through four frozen feature extractors, and produces:

- **Activity labels** — what activities are happening (multi-label)
- **Temporal segments** — when each activity starts and ends
- **Scene graph** — which objects are present and how they relate

The core model is a 4-layer Transformer encoder (~3.4M trainable parameters). All visual backbones (CLIP ViT-B/16, DETR) are frozen.

---

## Prerequisites

| Requirement | Details |
|---|---|
| OS | Ubuntu / any Linux |
| Python | 3.10+ |
| Storage | ~55 GB for raw data + ~5 GB for extracted features |
| Google account | For Colab GPU (feature extraction + training) |
| Internet | Required during feature extraction |

---

## Setup

### 1 — Clone and install

```bash
git clone https://github.com/BRshreya29/ee508.git
cd ee508
pip install -r requirements.txt
```

### 2 — Get the data

You need the Action Genome annotations and Charades videos.

**Option A — Data inside the repo (everything in one place)**

Create the folders and place data directly:

```bash
mkdir -p data/action_genome/annotations
mkdir -p data/charades_videos/Charades_v1_480
mkdir -p features checkpoints
```

```
ee508/
├── data/
│   ├── action_genome/annotations/
│   │   ├── object_bbox_and_relationship.pkl   (~130 MB)
│   │   ├── person_bbox.pkl                    (~149 MB)
│   │   ├── object_classes.txt
│   │   ├── relationship_classes.txt
│   │   └── frame_list.txt
│   └── charades_videos/Charades_v1_480/
│       ├── 001YG.mp4
│       └── ...                                (~50 GB)
```

No symlinks needed. Run all scripts from inside `ee508/` and relative paths resolve correctly.

**Option B — Data on a separate disk (original setup)**

Place data on your large disk, then symlink into the repo:

```bash
cd ~/path/to/ee508
ln -s /media/yourname/disk/ee508_data/charades_videos data/charades_videos
ln -s /media/yourname/disk/ee508_data/action_genome  data/action_genome
mkdir -p features checkpoints
```

---

## Running the Pipeline

### Step 1 — Filter and sample (local, CPU, fast)

```bash
# Filter annotations to 8-object / 10-predicate taxonomy
python3 scripts/filter_annotations.py
# → data/filtered_annotations.json  (~3728 videos retained)

# Sample 32 frames per video
python3 scripts/sample_frames.py
# → data/frame_indices.json
```

### Step 2 — Feature extraction (Colab GPU)

Raw videos stay on your disk. A tunnel streams them to Colab.

**Start the tunnel — keep this terminal open:**

```bash
bash scripts/serve_videos.sh
```

Wait for:
```
https://xxxx.trycloudflare.com
```

Copy that URL.

> If you see QUIC/UDP errors, the script already forces `--protocol http2`. Re-run once.

**Run the Colab notebook:**

1. Open [colab.research.google.com](https://colab.research.google.com)
2. Upload `EE508_Colab.ipynb`
3. Set **Runtime → Change runtime type → T4 GPU**
4. Upload the project folder to `MyDrive/ee508/` via the file browser
5. Upload `data/frame_indices.json` to `MyDrive/ee508/data/`
6. Paste your tunnel URL into the `TUNNEL_URL` variable in cell 2
7. Run all cells

Features are saved to `MyDrive/ee508/features/`. Each video gets four files:

```
features/{VIDEO_ID}/
├── clip.pt         [32, 512]   CLIP ViT-B/16 tokens
├── framediff.pt    [32, 49]    motion features
├── objects.pt      [32, 5, 324] DETR object tokens
└── scene.pt        [32, 6]     scene MLP logits
```

**Download features back to your machine:**

```bash
# First time only
curl https://rclone.org/install.sh | sudo bash
rclone config   # add Google Drive remote, name it 'gdrive'

# Download
rclone copy gdrive:ee508/features  ~/path/to/ee508/features/  --progress
rclone copy gdrive:ee508/checkpoints/scene_mlp.pt \
           ~/path/to/ee508/checkpoints/scene_mlp.pt
```

### Step 3 — Prepare labels and train

```bash
python3 scripts/prepare_labels.py   # → data/labels.json
python3 scripts/split_dataset.py    # → data/split.json  (70/15/15)

python3 train.py                    # saves to checkpoints/
```

Training can also run in the Colab notebook (Step 6 of the notebook) if you prefer GPU training.

### Step 4 — Demo

```bash
python3 demo.py \
    --features_dir features/001YG \
    --checkpoint   checkpoints/best_map_model.pt \
    --out_dir      demo_output
```

Outputs three files in `demo_output/`:

| File | What it shows |
|---|---|
| `activity_timeline.png` | Predicted activities and their time extents |
| `scene_graph.png` | Object-relation graph for a selected frame |
| `attention_heatmap.png` | Cross-attention weights [32 × 192] — which objects the model attended to |

---

## Verify the Setup

```bash
# Unit tests — no data needed, runs on synthetic tensors
python3 -m pytest tests/ -v
# Expected: 24/24 passed

# End-to-end smoke test — synthetic data
python3 scripts/smoke_test.py
# Expected: "All smoke tests passed!"
```

---

## Repository Layout

```
ee508/
├── data/
│   ├── action_genome/          ← annotations (symlink or direct)
│   ├── charades_videos/        ← raw videos  (symlink or direct)
│   ├── filtered_annotations.json
│   ├── frame_indices.json
│   ├── labels.json
│   └── split.json
├── features/                   ← extracted .pt files (after Colab step)
│   └── {VIDEO_ID}/
│       ├── clip.pt  framediff.pt  objects.pt  scene.pt
├── checkpoints/                ← saved during training
├── scripts/
│   ├── serve_videos.sh         tunnel for Colab extraction
│   ├── filter_annotations.py   Step 1a
│   ├── sample_frames.py        Step 1b
│   ├── extract_*.py            Step 2 (run inside Colab)
│   ├── prepare_labels.py       Step 3a
│   ├── split_dataset.py        Step 3b
│   └── smoke_test.py           sanity check
├── src/                        model architecture
├── train.py
├── demo.py
├── video_demo/                 ← sample demo recordings
│   ├── demo1.webm
│   └── demo2.webm
├── EE508_Colab.ipynb
├── requirements.txt
└── tests/
```

---

## Architecture at a Glance

```
Video → sample 32 frames
           │
    ┌──────┴───────────────────────────────┐
    │  Four frozen extractors              │
    │  CLIP ViT-B/16   → video token       │
    │  Frame diff      → motion token      │
    │  DETR            → 5 object tokens   │
    │  Scene MLP       → scene token       │
    └──────────────────────────────────────┘
           │  256 tokens total (32 × 8)
    ┌──────┴───────────────────────────────┐
    │  Temporal Transformer (4L × 4H)      │
    │  → Cross-attention fusion            │
    │    Q = temporal  ·  KV = objects     │
    └──────────────────────────────────────┘
           │
    ┌──────┴──────┬──────────────┐
    │             │              │
 Activity   Localization   Scene graph
 (mAP)       (mIoU)        (Recall@20)
```

Trainable parameters: ~3.14M. All backbone weights stay frozen.

Best checkpoint: `checkpoints/best_map_model.pt` — mAP **21.37%** on test set (328 videos).

---

## Video Demos

Two sample inference recordings are in [`video_demo/`](video_demo/):

| File | Description |
|---|---|
| `demo1.webm` | Model inference on a kitchen/living-room activity clip |
| `demo2.webm` | Model inference on a second home-activity clip |

Each demo shows the three outputs produced by `demo.py`:
- **Activity timeline** — predicted activities with temporal extents
- **Scene graph** — object nodes and predicted spatial relations
- **Attention heatmap** — which objects the model attends to per frame

To run inference on your own video's pre-extracted features:

```bash
python3 demo.py \
    --features_dir features/<VIDEO_ID> \
    --checkpoint   checkpoints/best_map_model.pt \
    --out_dir      demo_output/<VIDEO_ID>
```
