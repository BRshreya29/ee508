"""Stage B2 — Temporal Transformer Encoder.

Reference: doc 07 — Temporal Transformer.
- 4-layer vanilla Transformer encoder
- 4 attention heads, D=256, FF dim=512
- Pre-norm (norm_first=True), GELU activation
- Dropout 0.1
"""
import torch
import torch.nn as nn

TOKENS_PER_FRAME = 8
K = 5  # object tokens per frame


class TemporalTransformerEncoder(nn.Module):
    """Vanilla Transformer encoder for temporal reasoning over the 256-token sequence."""

    def __init__(
        self,
        d_model=256,
        num_heads=4,
        num_layers=4,
        ff_dim=512,
        dropout=0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm: LayerNorm before attention
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),  # Final layer norm
        )

    def forward(self, tokens, src_key_padding_mask=None):
        """
        tokens: [B, 256, 256]  (batch, seq_len, d_model)
        src_key_padding_mask: [B, 256] — True where token should be ignored

        Returns: [B, 256, 256] — contextualized token sequence
        """
        return self.encoder(tokens, src_key_padding_mask=src_key_padding_mask)


def build_padding_mask(obj_mask, num_frames=32, tokens_per_frame=TOKENS_PER_FRAME, k=K):
    """Build full sequence padding mask from object padding mask.

    Args:
        obj_mask: [B, 32, 5] — True where object slot is padded
        num_frames: T
        tokens_per_frame: 8
        k: number of object tokens per frame

    Returns:
        [B, 256] — True where token should be ignored by attention
    """
    B, T, K_actual = obj_mask.shape
    device = obj_mask.device

    # Full mask: [B, T * tokens_per_frame]
    full_mask = torch.zeros(B, T * tokens_per_frame, dtype=torch.bool, device=device)

    for t in range(T):
        for k_idx in range(K_actual):
            # Object tokens are at positions 2..6 within each frame group
            token_position = t * tokens_per_frame + 2 + k_idx
            full_mask[:, token_position] = obj_mask[:, t, k_idx]

    return full_mask
