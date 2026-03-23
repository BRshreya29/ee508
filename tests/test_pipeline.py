"""Unit tests for pipeline components."""
import sys
import os
import json
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Test annotation filtering ──────────────────────────────────────────────

class TestFilterAnnotations:
    def setup_method(self):
        from scripts.filter_annotations import filter_frame_annotations, filter_video_annotations
        self.filter_frame = filter_frame_annotations
        self.filter_video = filter_video_annotations

    def test_keeps_valid_objects(self):
        frame = {
            "objects": [
                {"class": "person", "id": 1},
                {"class": "chair", "id": 2},
            ],
            "relations": [],
        }
        result = self.filter_frame(frame)
        assert result is not None
        assert len(result["objects"]) == 2

    def test_filters_out_invalid_objects(self):
        frame = {
            "objects": [
                {"class": "horse", "id": 1},
                {"class": "airplane", "id": 2},
            ],
            "relations": [],
        }
        result = self.filter_frame(frame)
        assert result is None

    def test_filters_invalid_predicates(self):
        frame = {
            "objects": [
                {"class": "person", "id": 1},
                {"class": "chair", "id": 2},
            ],
            "relations": [
                {"subject_id": 1, "object_id": 2, "predicate": "sitting on"},
                {"subject_id": 1, "object_id": 2, "predicate": "flying over"},
            ],
        }
        result = self.filter_frame(frame)
        assert len(result["relations"]) == 1

    def test_video_50_percent_threshold(self):
        video = {
            "frames": {
                "0": {"objects": [{"class": "person", "id": 1}], "relations": []},
                "1": {"objects": [{"class": "horse", "id": 2}], "relations": []},
            }
        }
        result = self.filter_video(video)
        assert result is not None  # 50% survive

    def test_video_below_threshold(self):
        video = {
            "frames": {
                "0": {"objects": [{"class": "horse", "id": 1}], "relations": []},
                "1": {"objects": [{"class": "horse", "id": 2}], "relations": []},
                "2": {"objects": [{"class": "person", "id": 3}], "relations": []},
            }
        }
        result = self.filter_video(video)
        assert result is None  # only 33% survive


# ── Test frame sampling ────────────────────────────────────────────────────

class TestFrameSampling:
    def setup_method(self):
        from scripts.sample_frames import sample_frame_indices
        self.sample = sample_frame_indices

    def test_standard_video(self):
        indices = self.sample(1000, 32)
        assert len(indices) == 32
        assert indices[0] == 0
        assert indices[-1] == 999

    def test_short_video(self):
        indices = self.sample(10, 32)
        assert len(indices) == 32
        assert indices[-1] == 9  # padded with last frame

    def test_exact_32_frames(self):
        indices = self.sample(32, 32)
        assert len(indices) == 32
        np.testing.assert_array_equal(indices, np.arange(32))

    def test_one_frame_video(self):
        indices = self.sample(1, 32)
        assert len(indices) == 32
        assert all(i == 0 for i in indices)


# ── Test label preparation ─────────────────────────────────────────────────

class TestLabelPreparation:
    def setup_method(self):
        from scripts.prepare_labels import (
            prepare_activity_labels,
            prepare_temporal_labels,
            prepare_scene_graph_labels,
        )
        self.prep_activity = prepare_activity_labels
        self.prep_temporal = prepare_temporal_labels
        self.prep_sg = prepare_scene_graph_labels

    def test_activity_labels(self):
        video = {"activities": [{"class_id": 5}, {"class_id": 10}]}
        labels = self.prep_activity(video)
        assert 5 in labels
        assert 10 in labels

    def test_temporal_labels(self):
        video = {
            "activities": [{"class_id": 0, "start": 5.0, "end": 15.0}],
            "duration": 30.0,
        }
        temporal = self.prep_temporal(video)
        assert 0 in temporal
        assert abs(temporal[0][0] - 5.0 / 30.0) < 1e-6
        assert abs(temporal[0][1] - 15.0 / 30.0) < 1e-6

    def test_sg_labels(self):
        video = {
            "frames": {
                "0": {
                    "objects": [],
                    "relations": [{
                        "subject_class": "person",
                        "object_class": "chair",
                        "predicate": "sitting on",
                    }],
                }
            }
        }
        sg = self.prep_sg(video)
        assert "0" in sg
        assert len(sg["0"]) == 1
        assert sg["0"][0][1] == 1  # "sitting on" is index 1

    def test_empty_video(self):
        labels = self.prep_activity({})
        assert labels == []


# ── Test dataset ───────────────────────────────────────────────────────────

class TestDataset:
    def test_dataset_getitem(self):
        """Test dataset with synthetic data."""
        import torch
        from src.dataset import ActionGenomeFeatureDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create synthetic features
            feat_dir = os.path.join(tmpdir, "features", "vid001")
            os.makedirs(feat_dir)
            torch.save(torch.randn(32, 512), os.path.join(feat_dir, "clip.pt"))
            torch.save(torch.randn(32, 49), os.path.join(feat_dir, "framediff.pt"))
            torch.save(torch.randn(32, 5, 324), os.path.join(feat_dir, "objects.pt"))
            torch.save(torch.randn(32, 6), os.path.join(feat_dir, "scene.pt"))

            # Create split and labels
            split = {"train": ["vid001"], "val": [], "test": []}
            labels = {"vid001": {"activity": [1, 5], "temporal": {"1": [0.1, 0.5]}, "scene_graph": {}}}

            split_path = os.path.join(tmpdir, "split.json")
            labels_path = os.path.join(tmpdir, "labels.json")
            with open(split_path, "w") as f:
                json.dump(split, f)
            with open(labels_path, "w") as f:
                json.dump(labels, f)

            dataset = ActionGenomeFeatureDataset(
                split_path, labels_path,
                features_root=os.path.join(tmpdir, "features"),
                split="train",
            )
            assert len(dataset) == 1

            clip, diff, obj, scene, mask, lbls = dataset[0]
            assert clip.shape == (32, 512)
            assert diff.shape == (32, 49)
            assert obj.shape == (32, 5, 324)
            assert scene.shape == (32, 6)
            assert mask.shape == (32, 5)
            assert lbls["activity"].shape == (157,)
            assert lbls["temporal"].shape == (32, 2)
            assert lbls["scene_graph"].shape == (32, 20)

    def test_collate_fn(self):
        import torch
        from src.dataset import collate_fn

        batch = []
        for _ in range(4):
            clip = torch.randn(32, 512)
            diff = torch.randn(32, 49)
            obj = torch.randn(32, 5, 324)
            scene = torch.randn(32, 6)
            mask = torch.zeros(32, 5, dtype=torch.bool)
            labels = {
                "activity": torch.zeros(157),
                "temporal": torch.zeros(32, 2),
                "activity_mask": torch.zeros(32, dtype=torch.bool),
                "scene_graph": torch.full((32, 20), -1, dtype=torch.long),
            }
            batch.append((clip, diff, obj, scene, mask, labels))

        c, d, o, s, m, l = collate_fn(batch)
        assert c.shape == (4, 32, 512)
        assert d.shape == (4, 32, 49)
        assert o.shape == (4, 32, 5, 324)
        assert s.shape == (4, 32, 6)
        assert l["activity"].shape == (4, 157)
