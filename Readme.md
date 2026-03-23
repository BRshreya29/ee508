# EE508 — Project Handoff Guide

## Prerequisites

| Requirement | Details |
|---|---|
| OS | Ubuntu (or any Linux) |
| Python | 3.10+ |
| Storage | ~5 GB local free (for features); raw videos (~50 GB) on a separate disk |
| Google account | For Colab GPU access |
| Internet | Required during feature extraction |

---

## 1 — Get the Code

```bash
# Copy the ee508/ project folder to your machine, then:
cd ~/path/to/ee508
pip install -r requirements.txt
```

---

## 2 — Get the Data

You need the Action Genome annotations and Charades videos.  
Place them on a large disk (e.g., `/media/yourname/Elements/ee508_data/`):

```
ee508_data/
├── action_genome/
│   └── annotations/
│       ├── object_bbox_and_relationship.pkl   (~130 MB)
│       ├── person_bbox.pkl                    (~149 MB)
│       ├── object_classes.txt
│       ├── relationship_classes.txt
│       └── frame_list.txt
└── charades_videos/
    └── Charades_v1_480/
        ├── 001YG.mp4
        └── ... (~50 GB)
```

**Create symlinks** so the project finds the data:
```bash
cd ~/path/to/ee508
ln -s /media/yourname/Elements/ee508_data/charades_videos data/charades_videos
ln -s /media/yourname/Elements/ee508_data/action_genome  data/action_genome
mkdir -p features
```

---

## 3 — Run the Data Pipeline (local, CPU)

These steps are fast and run locally:

```bash
cd ~/path/to/ee508

# Step 2: Filter annotations to 8-class / 10-predicate taxonomy
python3 scripts/filter_annotations.py
# Output: data/filtered_annotations.json  (~3728 videos retained)

# Step 3: Sample 32 frames per video
python3 scripts/sample_frames.py
# Output: data/frame_indices.json
```

---

## 4 — Feature Extraction on Colab GPU

Raw video files stay on your disk. A tunnel streams them to Colab for GPU extraction.

### 4a — Start the local tunnel (keep this terminal open)

```bash
cd ~/path/to/ee508
bash scripts/serve_videos.sh
```

Wait for the line:
```
https://xxxx.trycloudflare.com
```
**Copy this URL** — you'll paste it into Colab.

> If you see QUIC/UDP errors, the script already forces `--protocol http2` (TCP). Re-run once if needed.

### 4b — Run Colab notebook

1. Open [colab.research.google.com](https://colab.research.google.com)
2. Upload [EE508_Colab.ipynb](file:///home/shreya/studies/project/ee508/EE508_Colab.ipynb)
3. Set **Runtime → Change runtime type → T4 GPU**
4. Upload the project folder to `MyDrive/ee508/` via the Colab file browser
5. Paste your tunnel URL into the `TUNNEL_URL` variable in cell 2
6. Also upload `data/frame_indices.json` to `MyDrive/ee508/data/`
7. Run all cells — features are saved to `MyDrive/ee508/features/`

The notebook extracts:
- `clip.pt` — CLIP ViT-B/16 tokens `[32, 512]`
- `framediff.pt` — motion features `[32, 49]`
- `objects.pt` — DETR object tokens `[32, 5, 324]`
- `scene.pt` — scene MLP logits `[32, 6]`

### 4c — Download features back to your PC

```bash
# Install rclone (first time only)
curl https://rclone.org/install.sh | sudo bash
rclone config   # add Google Drive remote, name it 'gdrive'

# Download
rclone copy gdrive:ee508/features ~/path/to/ee508/features/ --progress
rclone copy gdrive:ee508/checkpoints/scene_mlp.pt \
           ~/path/to/ee508/checkpoints/scene_mlp.pt
```

---

## 5 — Prepare Labels & Train

```bash
cd ~/path/to/ee508

python3 scripts/prepare_labels.py     # → data/labels.json
python3 scripts/split_dataset.py      # → data/split.json  (70/15/15 split)

python3 train.py                      # trains on CPU, saves checkpoints/
```

> Training on CPU is slow but feasible. Alternatively, run [train.py](file:///home/shreya/studies/project/ee508/train.py) on Colab too (see the notebook's Step 6).

---

## 6 — Run Demo

```bash
python3 demo.py \
    --features_dir features/001YG \
    --checkpoint checkpoints/best_model.pt \
    --out_dir demo_output
```

Produces 3 plots in `demo_output/`:
- `activity_timeline.png` — predicted activities over time
- `scene_graph.png` — object-relation graph for a frame
- `attention_heatmap.png` — cross-attention weights `[32 × 192]`

---

## 7 — Verify Everything Works

```bash
# Run all unit tests (no data needed, uses synthetic tensors)
python3 -m pytest tests/ -v
# Expected: 36/36 passed

# End-to-end smoke test (synthetic data)
python3 scripts/smoke_test.py
# Expected: "All smoke tests passed!"
```

---

## File Map

```
ee508/
├── data/
│   ├── action_genome   → symlink to hard disk
│   ├── charades_videos → symlink to hard disk
│   ├── filtered_annotations.json   (Step 2 output)
│   ├── frame_indices.json          (Step 3 output)
│   ├── labels.json                 (Step 5 output)
│   └── split.json                  (Step 5 output)
├── features/           ← downloaded from Drive after Colab extraction
│   └── {VIDEO_ID}/
│       ├── clip.pt  framediff.pt  objects.pt  scene.pt
├── checkpoints/        ← saved during training
├── scripts/
│   ├── serve_videos.sh             # tunnel for Colab extraction
│   ├── filter_annotations.py       # Step 2
│   ├── sample_frames.py            # Step 3
│   ├── extract_*.py                # Step 4 (run in Colab)
│   ├── prepare_labels.py           # Step 5
│   └── split_dataset.py            # Step 5
├── src/                # model architecture
├── train.py            # training loop
├── demo.py             # visualization
├── EE508_Colab.ipynb   # Colab notebook
└── tests/              # unit + model tests
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `cloudflared` QUIC timeout | Script already uses `--protocol http2`. Re-run |
| `FileExistsError` on symlink | `os.unlink('/content/ee508')` then re-run cell |
| Videos not found | Check symlink: `ls -la data/charades_videos/` |
| [filter_annotations.py](file:///home/shreya/studies/project/ee508/scripts/filter_annotations.py) empty output | Annotations must be `.pkl` files in `data/action_genome/annotations/` |
| OOM during DETR extraction on Colab | Reduce batch size or restart runtime and resume from last checkpoint |
