"""Stage B1 — Linear Projection & Temporal Positional Encoding.

Reference: doc 06 — Linear Projection.
- Four independent linear layers mapping each stream to D=256
- Learned [PAD] token for missing object detections
- Learned frame-index positional encoding (32 positions)
"""
import torch
import torch.nn as nn


class ProjectionBlock(nn.Module):
    """Projects all four feature streams to the shared D=256 embedding space
    and assembles them into a single token sequence."""

    def __init__(self, d_model=256):
        super().__init__()
        self.d_model = d_model

        # Stream projections (doc 06 table)
        self.video_proj = nn.Linear(512, d_model)   # Stream 1: CLIP CLS
        self.motion_proj = nn.Linear(49, d_model)    # Stream 2: frame diff
        self.object_proj = nn.Linear(324, d_model)   # Stream 3: DETR objects
        self.scene_proj = nn.Linear(6, d_model)      # Stream 4: scene probs

        # Learned [PAD] token for missing object detections (doc 04)
        self.pad_token = nn.Parameter(torch.zeros(d_model))

    def forward(self, clip_feat, diff_feat, obj_feat, scene_feat, obj_mask):
        """
        clip_feat:  [B, 32, 512]
        diff_feat:  [B, 32, 49]
        obj_feat:   [B, 32, 5, 324]
        scene_feat: [B, 32, 6]
        obj_mask:   [B, 32, 5]  — True where object slot is padding

        Returns: [B, 256, 256] — (batch, seq_len, d_model)
        """
        B, T = clip_feat.shape[0], clip_feat.shape[1]
        K = obj_feat.shape[2]

        v = self.video_proj(clip_feat)      # [B, 32, 256]
        m = self.motion_proj(diff_feat)     # [B, 32, 256]
        s = self.scene_proj(scene_feat)     # [B, 32, 256]

        # Object tokens: [B, 32, 5, 256]
        o = self.object_proj(obj_feat)

        # Replace padded object slots with learned pad token
        pad = self.pad_token.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 256]
        pad = pad.expand(B, T, K, self.d_model)
        o = torch.where(obj_mask.unsqueeze(-1), pad, o)

        # Assemble per-frame token groups: [v | m | o1..o5 | s]
        tokens = torch.cat([
            v.unsqueeze(2),   # [B, 32, 1, 256]
            m.unsqueeze(2),   # [B, 32, 1, 256]
            o,                # [B, 32, 5, 256]
            s.unsqueeze(2),   # [B, 32, 1, 256]
        ], dim=2)             # [B, 32, 8, 256]

        # Flatten to sequence: [B, 256, 256]
        tokens = tokens.view(B, T * 8, self.d_model)
        return tokens


class TemporalPositionalEncoding(nn.Module):
    """Learned frame-index positional encoding.

    All 8 tokens within the same frame receive the same positional encoding.
    """

    def __init__(self, num_frames=32, tokens_per_frame=8, d_model=256):
        super().__init__()
        self.frame_embeddings = nn.Embedding(num_frames, d_model)
        self.tokens_per_frame = tokens_per_frame

    def forward(self, tokens):
        """
        tokens: [B, T*tokens_per_frame, d_model]
        Returns: tokens + positional encoding, same shape
        """
        B, seq_len, d = tokens.shape
        T = seq_len // self.tokens_per_frame

        # Frame indices: [0,0,...,0, 1,1,...,1, ..., 31,31,...,31]
        frame_indices = torch.arange(T, device=tokens.device)
        frame_indices = frame_indices.repeat_interleave(self.tokens_per_frame)  # [256]
        frame_indices = frame_indices.unsqueeze(0).expand(B, -1)  # [B, 256]

        pe = self.frame_embeddings(frame_indices)  # [B, 256, d_model]
        return tokens + pe
