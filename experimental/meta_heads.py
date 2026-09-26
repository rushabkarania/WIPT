"""Embedding-only episodic heads for MAML-inspired target-support adaptation."""

from __future__ import annotations

import torch
import torch.nn as nn

from configs.experiment import EMBED_DIM, WIPT_DROPOUT, WIPT_HEADS, WIPT_LAYERS


class MetaTransformerLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        num_heads: int = WIPT_HEADS,
        dropout: float = WIPT_DROPOUT,
        mlp_ratio: float = 4.0,
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

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        residual = tokens
        x = self.norm1(tokens)
        attended, _ = self.attn(x, x, x, need_weights=False)
        tokens = residual + attended
        return tokens + self.mlp(self.norm2(tokens))


class MetaEpisodicHead(nn.Module):
    """Euclidean prototype head with support-only or joint query/support transformation."""

    def __init__(
        self,
        adaptation: str = "joint",
        *,
        embed_dim: int = EMBED_DIM,
        num_layers: int = WIPT_LAYERS,
        num_heads: int = WIPT_HEADS,
        dropout: float = WIPT_DROPOUT,
    ):
        super().__init__()
        if adaptation not in {"joint", "support"}:
            raise ValueError("adaptation must be 'joint' or 'support'")
        self.adaptation = adaptation
        self.layers = nn.ModuleList(
            [
                MetaTransformerLayer(embed_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def _transform_support(self, support: torch.Tensor) -> torch.Tensor:
        x = support.unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x).squeeze(0)

    def _transform_joint(
        self, support: torch.Tensor, query: torch.Tensor, n_way: int, n_shot: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Single-query processing is kept here deliberately: this branch answers the
        # support-supervised test-time adaptation question separately from q>1 context.
        n_query = query.shape[0]
        tokens = torch.cat(
            [query.unsqueeze(1), support.unsqueeze(0).expand(n_query, -1, -1)], dim=1
        )
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)
        return tokens[:, 0], tokens[:, 1:].view(n_query, n_way, n_shot, -1)

    def forward(
        self, support: torch.Tensor, query: torch.Tensor, n_way: int, n_shot: int
    ) -> torch.Tensor:
        if self.adaptation == "support":
            transformed_support = self._transform_support(support)
            prototypes = transformed_support.view(n_way, n_shot, -1).mean(1)
            return -torch.cdist(query, prototypes)
        transformed_query, transformed_support = self._transform_joint(
            support, query, n_way, n_shot
        )
        prototypes = transformed_support.mean(2)
        return -torch.linalg.vector_norm(
            transformed_query.unsqueeze(1) - prototypes, dim=-1
        )
