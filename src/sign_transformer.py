#!/usr/bin/env python3
"""
SignTransformer — Vision Transformer for sign language recognition.

Architecture (inspired by Vaswani et al. 2017 "Attention Is All You Need"):
    1. Linear projection:       input_dim (156) → d_model (256)
    2. Learnable positional encoding: 30 temporal positions
    3. N × Transformer encoder layers:
           Multi-head self-attention (8 heads)
         + Feed-forward network (256 → 512 → 256)
         + Pre-norm LayerNorm + residual connections
         + Dropout (0.1)
    4. Global average pooling over 30 output tokens
    5. LayerNorm → Linear(d_model, num_classes)

Each frame's 156-dim MediaPipe vector is treated as one "token",
analogous to a word embedding in NLP. Self-attention lets every
frame attend to every other frame, learning which moments of the
sign carry the most discriminative information.

Importable with no side effects.
"""

import torch
import torch.nn as nn


class SignTransformer(nn.Module):
    """
    Temporal Vision Transformer for sign language classification.

    Args:
        input_dim   : feature dimension per frame (156 for MediaPipe pose)
        d_model     : internal transformer dimension (256)
        n_heads     : number of attention heads (8)
        n_layers    : number of encoder layers (4)
        d_ff        : feed-forward hidden dimension (512)
        dropout     : dropout probability (0.1)
        num_classes : number of output classes
        max_seq_len : maximum sequence length (30 frames)
    """

    def __init__(
        self,
        input_dim: int = 156,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        num_classes: int = 8,
        max_seq_len: int = 30,
    ):
        super().__init__()

        self.d_model = d_model

        # ── Input projection: 156 → d_model ──────────────────────────────────
        self.input_proj = nn.Linear(input_dim, d_model)

        # ── Learnable positional encoding ─────────────────────────────────────
        # Shape: (1, max_seq_len, d_model) — broadcast over batch
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        # ── Transformer encoder ───────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,   # input shape: (batch, seq, features)
            norm_first=True,    # pre-norm: more stable with small datasets
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        # ── Classification head ───────────────────────────────────────────────
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.input_proj.weight, std=0.02)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)  e.g. (32, 30, 156)
        Returns:
            logits: (batch, num_classes)
        """
        B, T, _ = x.shape

        # Project each frame's feature vector into d_model space
        x = self.input_proj(x)                      # (B, T, d_model)

        # Add positional encoding so the model knows frame order
        x = x + self.pos_embed[:, :T, :]            # (B, T, d_model)

        # Self-attention: every frame attends to every other frame
        x = self.encoder(x)                         # (B, T, d_model)

        # Global average pooling — aggregate all 30 frame representations
        x = x.mean(dim=1)                           # (B, d_model)

        # Classify
        x = self.norm(x)
        x = self.drop(x)
        return self.head(x)                         # (B, num_classes)
