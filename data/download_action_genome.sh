#!/usr/bin/env bash
# Download Action Genome annotations and Charades videos
# Reference: doc 01 — Step 1
set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Action Genome annotations ==="
echo "Download from: https://www.actiongenome.org/#download"
echo "Place annotation files into: ${DATA_DIR}/action_genome/annotations/"
echo ""
echo "Required files:"
echo "  - object_bbox_and_relationship.pkl  (or .json)"
echo "  - person_bbox.pkl"
echo "  - frame_list.txt"
echo ""

echo "=== Charades videos ==="
echo "Download from: https://prior.allenai.org/projects/charades"
echo "Place .mp4 files into: ${DATA_DIR}/charades_videos/"
echo ""
echo "Estimated total download: ~50 GB"
echo ""

# Create target directories
mkdir -p "${DATA_DIR}/action_genome/annotations"
mkdir -p "${DATA_DIR}/charades_videos"

echo "Directories created. Please download data manually from the URLs above."
echo "After downloading, run: python scripts/filter_annotations.py"
