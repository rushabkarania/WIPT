"""Refined WIPT variants reported in the ablation study."""

from __future__ import annotations

import torch
import torch.nn as nn

from configs.experiment import BACKBONE, WIPT_DROPOUT, WIPT_HEADS, WIPT_LAYERS
from models.backbone import build_frozen_encoder


class RefinedWIPTLayer(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, dropout: float, mlp_ratio: float = 4.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, tokens: torch.Tensor, attention_temperature: float = 1.0
    ) -> torch.Tensor:
        residual = tokens
        normalized = self.norm1(tokens)
        if attention_temperature == 1.0:
            attended, _ = self.attn(normalized, normalized, normalized)
        else:
            scale = 1.0 / attention_temperature
            attended, _ = self.attn(normalized * scale, normalized * scale, normalized)
        tokens = residual + attended
        return tokens + self.mlp(self.norm2(tokens))


class RefinedWIPT(nn.Module):
    def __init__(
        self,
        *,
        backbone: str = BACKBONE,
        pretrained: bool = True,
        num_layers: int = WIPT_LAYERS,
        num_heads: int = WIPT_HEADS,
        dropout: float = WIPT_DROPOUT,
        attention_temperature: float = 1.0,
        support_dropout: float = 0.0,
        input_norm: bool = False,
        residual_query: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = build_frozen_encoder(backbone, pretrained=pretrained)
        self.embed_dim = self.encoder.num_features
        self.attention_temperature = attention_temperature
        self.support_dropout = support_dropout
        self.residual_query = residual_query
        self.use_input_norm = input_norm
        self.input_layer_norm = nn.LayerNorm(self.embed_dim) if input_norm else None
        self.transformer_layers = nn.ModuleList(
            [
                RefinedWIPTLayer(self.embed_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(self.embed_dim)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def forward(
        self,
        support_images: torch.Tensor,
        query_images: torch.Tensor,
        n_way: int,
        n_shot: int,
    ):
        support = self.encode(support_images)
        query = self.encode(query_images)
        original_query = query

        if self.input_layer_norm is not None:
            support = self.input_layer_norm(support)
            query = self.input_layer_norm(query)

        if self.training and self.support_dropout > 0:
            keep = torch.bernoulli(
                torch.full(
                    (support.shape[0], 1),
                    1 - self.support_dropout,
                    device=support.device,
                )
            )
            support = support * keep

        n_query = query.shape[0]
        tokens = torch.cat(
            [query.unsqueeze(1), support.unsqueeze(0).expand(n_query, -1, -1)],
            dim=1,
        )
        for layer in self.transformer_layers:
            tokens = layer(tokens, self.attention_temperature)
        tokens = self.norm(tokens)

        transformed_query = tokens[:, 0]
        if self.residual_query:
            transformed_query = 0.5 * (transformed_query + original_query)
        transformed_support = tokens[:, 1:].view(n_query, n_way, n_shot, -1)
        prototypes = transformed_support.mean(dim=2)
        return -torch.cdist(transformed_query.unsqueeze(1), prototypes).squeeze(1)
