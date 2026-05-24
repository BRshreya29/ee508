"""Unit tests for model architecture — tensor shapes and model behavior."""
import sys
import os
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestProjectionBlock:
    def test_output_shape(self):
        from src.projection import ProjectionBlock
        proj = ProjectionBlock(d_model=256)
        B = 2
        clip = torch.randn(B, 32, 768)
        diff = torch.randn(B, 32, 49)
        obj = torch.randn(B, 32, 5, 324)
        scene = torch.randn(B, 32, 6)
        mask = torch.zeros(B, 32, 5, dtype=torch.bool)

        tokens, combined_mask = proj(clip, diff, obj, scene, mask)
        assert tokens.shape == (B, 256, 256), f"Expected (2, 256, 256), got {tokens.shape}"
        assert combined_mask.shape == (B, 32, 5), f"Expected combined_mask (2, 32, 5), got {combined_mask.shape}"

    def test_pad_token_applied(self):
        from src.projection import ProjectionBlock
        proj = ProjectionBlock(d_model=256)
        B = 1
        clip = torch.randn(B, 32, 768)
        diff = torch.randn(B, 32, 49)
        obj = torch.randn(B, 32, 5, 324)
        scene = torch.randn(B, 32, 6)
        mask = torch.ones(B, 32, 5, dtype=torch.bool)  # all padded

        tokens, combined_mask = proj(clip, diff, obj, scene, mask)
        assert tokens.shape == (1, 256, 256)
        # All slots padded → combined_mask should be all True
        assert combined_mask.all()

    def test_low_conf_gating(self):
        """Low-norm object features should be gated even without explicit mask."""
        from src.projection import ProjectionBlock
        proj = ProjectionBlock(d_model=256, conf_thresh_l2=50.0)  # very high threshold
        B = 1
        clip = torch.randn(B, 32, 768)
        diff = torch.randn(B, 32, 49)
        obj = torch.randn(B, 32, 5, 324) * 0.001  # near-zero → low norm
        scene = torch.randn(B, 32, 6)
        mask = torch.zeros(B, 32, 5, dtype=torch.bool)  # no explicit padding

        _tokens, combined_mask = proj(clip, diff, obj, scene, mask)
        # All objects have near-zero norm → should all be gated
        assert combined_mask.all(), "Low-norm objects should be masked by confidence gating"


class TestTemporalPositionalEncoding:
    def test_output_shape(self):
        from src.projection import TemporalPositionalEncoding
        pe = TemporalPositionalEncoding(num_frames=32, tokens_per_frame=8, d_model=256)
        tokens = torch.randn(2, 256, 256)
        out = pe(tokens)
        assert out.shape == (2, 256, 256)

    def test_adds_to_tokens(self):
        from src.projection import TemporalPositionalEncoding
        pe = TemporalPositionalEncoding(num_frames=32, tokens_per_frame=8, d_model=256)
        tokens = torch.zeros(1, 256, 256)
        out = pe(tokens)
        # Output should not be all zeros (PE adds non-zero embeddings)
        assert out.abs().sum() > 0


class TestTransformerEncoder:
    def test_output_shape(self):
        from src.transformer import TemporalTransformerEncoder
        enc = TemporalTransformerEncoder(d_model=256, num_heads=4, num_layers=4, ff_dim=512)
        tokens = torch.randn(2, 256, 256)
        out = enc(tokens)
        assert out.shape == (2, 256, 256)

    def test_with_padding_mask(self):
        from src.transformer import TemporalTransformerEncoder, build_padding_mask
        enc = TemporalTransformerEncoder(d_model=256, num_heads=4, num_layers=4, ff_dim=512)
        tokens = torch.randn(2, 256, 256)
        obj_mask = torch.zeros(2, 32, 5, dtype=torch.bool)
        obj_mask[:, :, 3:] = True  # last 2 objects padded
        padding_mask = build_padding_mask(obj_mask)
        assert padding_mask.shape == (2, 256)
        out = enc(tokens, src_key_padding_mask=padding_mask)
        assert out.shape == (2, 256, 256)


class TestCrossAttentionFusion:
    def test_output_shape(self):
        from src.cross_attention import CrossAttentionFusion
        ca = CrossAttentionFusion(d_model=256, num_heads=4)
        queries = torch.randn(2, 32, 256)
        kv_pool = torch.randn(2, 192, 256)
        fused, weights = ca(queries, kv_pool)
        assert fused.shape == (2, 32, 256)
        assert weights.shape == (2, 32, 192)

    def test_attention_weights_stored(self):
        from src.cross_attention import CrossAttentionFusion
        ca = CrossAttentionFusion(d_model=256, num_heads=4)
        queries = torch.randn(1, 32, 256)
        kv_pool = torch.randn(1, 192, 256)
        ca(queries, kv_pool)
        assert ca._last_attn_weights is not None
        assert ca._last_attn_weights.shape == (1, 32, 192)

    def test_with_padding_mask(self):
        from src.cross_attention import CrossAttentionFusion
        ca = CrossAttentionFusion(d_model=256, num_heads=4)
        queries = torch.randn(2, 32, 256)
        kv_pool = torch.randn(2, 192, 256)
        kv_mask = torch.zeros(2, 192, dtype=torch.bool)
        kv_mask[:, 150:160] = True
        fused, weights = ca(queries, kv_pool, key_padding_mask=kv_mask)
        assert fused.shape == (2, 32, 256)


class TestOutputHeads:
    def test_activity_head(self):
        from src.output_heads import ActivityClassificationHead
        head = ActivityClassificationHead(d_model=256, num_classes=157)
        fused = torch.randn(2, 32, 256)
        logits = head(fused)
        assert logits.shape == (2, 157)

    def test_localization_head(self):
        from src.output_heads import LocalizationHead
        head = LocalizationHead(d_model=256)
        fused = torch.randn(2, 32, 256)
        preds = head(fused)
        assert preds.shape == (2, 32, 2)
        assert preds.min() >= 0.0 and preds.max() <= 1.0

    def test_scene_graph_head(self):
        from src.output_heads import SceneGraphHead
        head = SceneGraphHead(d_model=256, num_relations=11)
        fused = torch.randn(2, 32, 256)
        obj_tokens = torch.randn(2, 32, 5, 256)
        logits = head(fused, obj_tokens)
        assert logits.shape == (2, 32, 20, 11)

    def test_scene_graph_head_with_bbox(self):
        """SceneGraphHead should accept optional bbox_coords and output same shape."""
        from src.output_heads import SceneGraphHead
        head = SceneGraphHead(d_model=256, num_relations=11)
        fused = torch.randn(2, 32, 256)
        obj_tokens = torch.randn(2, 32, 5, 256)
        bbox_coords = torch.rand(2, 32, 5, 4)  # (cx, cy, w, h) in [0, 1]
        logits = head(fused, obj_tokens, bbox_coords=bbox_coords)
        assert logits.shape == (2, 32, 20, 11)


class TestLossFunctions:
    def test_activity_loss(self):
        from src.output_heads import activity_loss
        logits = torch.randn(4, 157, requires_grad=True)
        targets = torch.zeros(4, 157)
        targets[:, 0] = 1.0
        loss = activity_loss(logits, targets)
        assert loss.requires_grad
        assert loss.item() > 0

    def test_focal_loss_down_weights_easy_negatives(self):
        """Focal loss should give lower loss to very confident correct predictions."""
        from src.output_heads import activity_loss
        # High-confidence correct prediction (logit=10 for target=1)
        logits_easy = torch.tensor([[10.0] + [0.0] * 156])
        targets_easy = torch.tensor([[1.0] + [0.0] * 156])
        # Low-confidence correct prediction (logit=0.1 for target=1)
        logits_hard = torch.tensor([[0.1] + [0.0] * 156])
        targets_hard = torch.tensor([[1.0] + [0.0] * 156])
        loss_easy = activity_loss(logits_easy, targets_easy)
        loss_hard = activity_loss(logits_hard, targets_hard)
        # Easy (confident) examples should have lower loss due to focal weighting
        assert loss_easy.item() < loss_hard.item(), (
            f"Focal loss should down-weight easy: easy={loss_easy:.4f} hard={loss_hard:.4f}"
        )

    def test_localization_loss(self):
        from src.output_heads import localization_loss
        preds = torch.randn(4, 32, 2).sigmoid()
        targets = torch.rand(4, 32, 2)
        mask = torch.zeros(4, 32, dtype=torch.bool)
        mask[:, 5:15] = True
        loss = localization_loss(preds, targets, mask)
        assert loss.item() >= 0

    def test_localization_loss_empty_mask(self):
        from src.output_heads import localization_loss
        preds = torch.randn(4, 32, 2).sigmoid()
        targets = torch.rand(4, 32, 2)
        mask = torch.zeros(4, 32, dtype=torch.bool)
        loss = localization_loss(preds, targets, mask)
        assert loss.item() == 0.0

    def test_scene_graph_loss(self):
        from src.output_heads import scene_graph_loss
        logits = torch.randn(4, 32, 20, 11, requires_grad=True)
        targets = torch.full((4, 32, 20), -1, dtype=torch.long)
        targets[:, :, 0] = 0  # set some valid targets
        loss = scene_graph_loss(logits, targets)
        assert loss.requires_grad

    def test_total_loss(self):
        from src.output_heads import total_loss
        act_logits = torch.randn(4, 157, requires_grad=True)
        loc_preds = torch.rand(4, 32, 2).requires_grad_(True)
        sg_logits = torch.randn(4, 32, 20, 11, requires_grad=True)
        act_targets = torch.zeros(4, 157)
        loc_targets = torch.rand(4, 32, 2)
        act_mask = torch.zeros(4, 32, dtype=torch.bool)
        act_mask[:, 5:15] = True
        sg_targets = torch.full((4, 32, 20), -1, dtype=torch.long)
        sg_targets[:, :, 0] = 0

        loss, components = total_loss(
            act_logits, loc_preds, sg_logits,
            act_targets, loc_targets, act_mask, sg_targets
        )
        assert loss.requires_grad
        assert "activity" in components
        assert "localization" in components
        assert "scene_graph" in components


class TestFullModel:
    def test_forward_pass(self):
        from src.model import SceneActivityModel
        model = SceneActivityModel(
            d_model=256, num_frames=32, tokens_per_frame=8, k=5,
            num_heads=4, num_transformer_layers=4, ff_dim=512,
            num_activity_classes=157, num_relations=11,
        )
        B = 2
        clip = torch.randn(B, 32, 768)
        diff = torch.randn(B, 32, 49)
        obj = torch.randn(B, 32, 5, 324)
        scene = torch.randn(B, 32, 6)
        mask = torch.zeros(B, 32, 5, dtype=torch.bool)

        act, loc, sg, attn = model(clip, diff, obj, scene, mask)

        assert act.shape == (B, 157), f"act shape: {act.shape}"
        assert loc.shape == (B, 32, 2), f"loc shape: {loc.shape}"
        assert sg.shape == (B, 32, 20, 11), f"sg shape: {sg.shape}"
        assert attn.shape == (B, 32, 192), f"attn shape: {attn.shape}"

    def test_gradient_flow(self):
        from src.model import SceneActivityModel
        from src.output_heads import total_loss

        model = SceneActivityModel()
        B = 2
        clip = torch.randn(B, 32, 768)
        diff = torch.randn(B, 32, 49)
        obj = torch.randn(B, 32, 5, 324)
        scene = torch.randn(B, 32, 6)
        mask = torch.zeros(B, 32, 5, dtype=torch.bool)

        act_logits, loc_preds, sg_logits, _ = model(clip, diff, obj, scene, mask)

        act_targets = torch.zeros(B, 157)
        loc_targets = torch.rand(B, 32, 2)
        act_mask = torch.ones(B, 32, dtype=torch.bool)
        sg_targets = torch.zeros(B, 32, 20, dtype=torch.long)

        loss, _ = total_loss(
            act_logits, loc_preds, sg_logits,
            act_targets, loc_targets, act_mask, sg_targets
        )
        loss.backward()

        # Check gradients exist for trainable components
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_parameter_count(self):
        from src.model import SceneActivityModel
        model = SceneActivityModel()
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Expected ~3.4M trainable parameters (doc 00)
        assert 2_000_000 < total < 5_000_000, f"Trainable params: {total:,}"

    def test_attention_weights_accessible(self):
        from src.model import SceneActivityModel
        model = SceneActivityModel()
        B = 1
        clip = torch.randn(B, 32, 768)
        diff = torch.randn(B, 32, 49)
        obj = torch.randn(B, 32, 5, 324)
        scene = torch.randn(B, 32, 6)
        mask = torch.zeros(B, 32, 5, dtype=torch.bool)

        model(clip, diff, obj, scene, mask)
        weights = model.get_attention_weights()
        assert weights is not None
        assert weights.shape == (1, 32, 192)
