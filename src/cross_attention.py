"""Cross-Attention Fusion.

- Q = temporal summary tokens (mean-pooled per frame) [B, 32, 256]
- KV = object + scene tokens from encoder output  [B, 192, 256]
- 4-head scaled dot-product attention
- _last_attn_weights stored for interpretability
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """Cross-attention: temporal queries attending to object + scene key-values."""

    def __init__(self, d_model=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # 64

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.dropout = nn.Dropout(dropout)

        # Stored for visualization — required by doc 08 and antigravity_rules
        self._last_attn_weights = None

    def forward(self, queries, kv_pool, key_padding_mask=None):
        """
        queries:          [B, 32, 256]   — temporal summary tokens
        kv_pool:          [B, 192, 256]  — object + scene tokens
        key_padding_mask: [B, 192]       — True where KV position is padded

        Returns:
            fused:   [B, 32, 256]   — spatially grounded context vectors
            weights: [B, 32, 192]   — attention weights (interpretability signal)
        """
        B, T, D = queries.shape
        S = kv_pool.shape[1]

        # Pre-norm on queries
        q = self.norm1(queries)
        k = kv_pool
        v = kv_pool

        # Project to multi-head Q, K, V
        Q = self.q_proj(q).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 32, 64]
        K = self.k_proj(k).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 192, 64]
        V = self.v_proj(v).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 192, 64]

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, 32, 192]

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, 192]
            attn_logits = attn_logits.masked_fill(mask, float("-inf"))

        attn_weights = attn_logits.softmax(dim=-1)  # [B, H, 32, 192]
        attn_weights_dropped = self.dropout(attn_weights)

        # Average across heads for interpretability storage
        self._last_attn_weights = attn_weights.mean(dim=1).detach()  # [B, 32, 192]

        # Weighted sum of values
        out = torch.matmul(attn_weights_dropped, V)  # [B, H, 32, 64]
        out = out.transpose(1, 2).contiguous()        # [B, 32, H, 64]
        out = out.view(B, T, D)                       # [B, 32, 256]
        out = self.out_proj(out)

        # Residual connection
        fused = queries + self.dropout(out)

        # Post-fusion FF (pre-norm)
        fused = fused + self.dropout(self.ff(self.norm2(fused)))

        return fused, self._last_attn_weights


def prepare_cross_attention_inputs(encoder_output, num_frames=32, tokens_per_frame=8, k=5):
    """Extract queries and KV pool from the Transformer encoder output.

    Args:
        encoder_output: [B, 256, 256] from Stage B2

    Returns:
        queries:  [B, 32, 256]  — temporal summary (mean over 8 tokens per frame)
        kv_pool:  [B, 192, 256] — all object + scene tokens
        kv_mask:  [B, 192]      — padding mask for KV positions
    """
    B = encoder_output.shape[0]
    D = encoder_output.shape[2]

    # Reshape: [B, 32, 8, 256]
    enc = encoder_output.view(B, num_frames, tokens_per_frame, D)

    # Queries: mean-pool over 8 tokens per frame → [B, 32, 256]
    queries = enc.mean(dim=2)

    # Extract object tokens (slots 2-6) and scene tokens (slot 7)
    object_tokens = enc[:, :, 2:2 + k, :]   # [B, 32, 5, 256]
    scene_tokens = enc[:, :, 7:8, :]         # [B, 32, 1, 256]

    # KV pool: concatenate flattened object + scene tokens
    kv_pool = torch.cat([
        object_tokens.reshape(B, num_frames * k, D),    # [B, 160, 256]
        scene_tokens.reshape(B, num_frames * 1, D),     # [B, 32, 256]
    ], dim=1)  # [B, 192, 256]

    return queries, kv_pool, object_tokens


def build_kv_padding_mask(obj_mask, num_frames=32, k=5):
    """Build padding mask for the KV pool.

    Args:
        obj_mask: [B, 32, 5] — True where object is padded

    Returns:
        [B, 192] — True where KV position is padded
    """
    B, T, K = obj_mask.shape

    # Object part: [B, 32*5] = [B, 160]
    obj_kv_mask = obj_mask.reshape(B, T * K)

    # Scene part: never padded → [B, 32]
    scene_kv_mask = torch.zeros(B, T, dtype=torch.bool, device=obj_mask.device)

    return torch.cat([obj_kv_mask, scene_kv_mask], dim=1)  # [B, 192]
