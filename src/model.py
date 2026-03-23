"""Full model wrapper — SceneActivityModel.

Assembles all stages:
  Stage B1: ProjectionBlock + TemporalPositionalEncoding
  Stage B2: TemporalTransformerEncoder
  Stage B3: CrossAttentionFusion
  Stage C:  ActivityClassificationHead, LocalizationHead, SceneGraphHead
"""
import torch
import torch.nn as nn

from .projection import ProjectionBlock, TemporalPositionalEncoding
from .transformer import TemporalTransformerEncoder, build_padding_mask
from .cross_attention import (
    CrossAttentionFusion,
    prepare_cross_attention_inputs,
    build_kv_padding_mask,
)
from .output_heads import (
    ActivityClassificationHead,
    LocalizationHead,
    SceneGraphHead,
)


class SceneActivityModel(nn.Module):
    """End-to-end model: projection → Transformer → cross-attention → 3 heads.

    Takes precomputed frozen features as input, produces all three outputs.
    """

    def __init__(
        self,
        d_model=256,
        num_frames=32,
        tokens_per_frame=8,
        k=5,
        num_heads=4,
        num_transformer_layers=4,
        ff_dim=512,
        dropout=0.1,
        num_activity_classes=157,
        num_relations=11,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_frames = num_frames
        self.tokens_per_frame = tokens_per_frame
        self.k = k

        # Stage B1 — Projection + Positional Encoding
        self.projection = ProjectionBlock(d_model=d_model)
        self.pos_encoding = TemporalPositionalEncoding(
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            d_model=d_model,
        )

        # Stage B2 — Temporal Transformer Encoder
        self.transformer = TemporalTransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        # Stage B3 — Cross-Attention Fusion
        self.cross_attention = CrossAttentionFusion(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Stage C — Output Heads
        self.activity_head = ActivityClassificationHead(
            d_model=d_model,
            num_classes=num_activity_classes,
            dropout=0.2,
        )
        self.localization_head = LocalizationHead(
            d_model=d_model,
            dropout=0.1,
        )
        self.scene_graph_head = SceneGraphHead(
            d_model=d_model,
            num_relations=num_relations,
            dropout=0.1,
        )

    def forward(self, clip_feat, diff_feat, obj_feat, scene_feat, obj_mask):
        """
        clip_feat:  [B, 32, 512]
        diff_feat:  [B, 32, 49]
        obj_feat:   [B, 32, 5, 324]
        scene_feat: [B, 32, 6]
        obj_mask:   [B, 32, 5] — True where object is padding

        Returns:
            act_logits:     [B, num_activity_classes]
            loc_preds:      [B, 32, 2]
            sg_logits:      [B, 32, 20, num_relations]
            attn_weights:   [B, 32, 192]
        """
        # Stage B1: project and add positional encoding
        tokens = self.projection(clip_feat, diff_feat, obj_feat, scene_feat, obj_mask)
        tokens = self.pos_encoding(tokens)  # [B, 256, 256]

        # Build padding mask for Transformer
        padding_mask = build_padding_mask(obj_mask)  # [B, 256]

        # Stage B2: Temporal Transformer
        encoder_output = self.transformer(tokens, src_key_padding_mask=padding_mask)

        # Stage B3: prepare queries/KV and run cross-attention
        queries, kv_pool, object_tokens = prepare_cross_attention_inputs(
            encoder_output,
            num_frames=self.num_frames,
            tokens_per_frame=self.tokens_per_frame,
            k=self.k,
        )
        kv_mask = build_kv_padding_mask(obj_mask, num_frames=self.num_frames, k=self.k)
        fused, attn_weights = self.cross_attention(queries, kv_pool, key_padding_mask=kv_mask)

        # Stage C: Output heads
        act_logits = self.activity_head(fused)                         # [B, 157]
        loc_preds = self.localization_head(fused)                      # [B, 32, 2]
        sg_logits = self.scene_graph_head(fused, object_tokens)        # [B, 32, 20, 11]

        return act_logits, loc_preds, sg_logits, attn_weights

    def get_attention_weights(self):
        """Access the stored cross-attention weights for interpretability."""
        return self.cross_attention._last_attn_weights
