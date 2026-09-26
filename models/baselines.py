"""Frozen-ViT few-shot baselines used in the paper."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.experiment import BACKBONE, WIPT_DROPOUT, WIPT_HEADS
from models.backbone import build_frozen_encoder


class CosineMatchingViT(nn.Module):
    """Cosine-matching control with class-wise support-similarity averaging."""

    def __init__(
        self,
        *,
        backbone: str = BACKBONE,
        pretrained: bool = True,
        encoder_state_dict: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = build_frozen_encoder(
            backbone,
            pretrained=pretrained,
            state_dict=encoder_state_dict,
        )
        self.embed_dim = self.encoder.num_features

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def forward(self, support_images, query_images, n_way, n_shot):
        support = F.normalize(self.encode(support_images), dim=1)
        query = F.normalize(self.encode(query_images), dim=1)
        similarity = query @ support.T
        labels = torch.arange(n_way, device=query.device).repeat_interleave(n_shot)
        scores = torch.empty(query.shape[0], n_way, device=query.device)
        for class_index in range(n_way):
            scores[:, class_index] = similarity[:, labels == class_index].mean(dim=1)
        return scores


class RelationHeadViT(nn.Module):
    def __init__(
        self,
        *,
        backbone: str = BACKBONE,
        pretrained: bool = True,
        relation_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = build_frozen_encoder(backbone, pretrained=pretrained)
        self.embed_dim = self.encoder.num_features
        self.relation = nn.Sequential(
            nn.Linear(self.embed_dim * 2, relation_dim),
            nn.ReLU(),
            nn.Linear(relation_dim, relation_dim // 2),
            nn.ReLU(),
            nn.Linear(relation_dim // 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def forward(self, support_images, query_images, n_way, n_shot):
        support = self.encode(support_images)
        query = self.encode(query_images)
        prototypes = support.view(n_way, n_shot, -1).mean(dim=1)
        query_pairs = query.unsqueeze(1).expand(-1, n_way, -1)
        prototype_pairs = prototypes.unsqueeze(0).expand(query.shape[0], -1, -1)
        pairs = torch.cat([query_pairs, prototype_pairs], dim=2)
        return self.relation(pairs.reshape(query.shape[0] * n_way, -1)).view(
            query.shape[0], n_way
        )


class SupportTransformerLayer(nn.Module):
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

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        residual = tokens
        normalized = self.norm1(tokens)
        attended, _ = self.attn(normalized, normalized, normalized)
        tokens = residual + attended
        return tokens + self.mlp(self.norm2(tokens))


class SupportTransformerViT(nn.Module):
    """Support-only Transformer control; the query is left untransformed."""

    def __init__(
        self,
        *,
        backbone: str = BACKBONE,
        pretrained: bool = True,
        num_layers: int = 2,
        num_heads: int = WIPT_HEADS,
        dropout: float = WIPT_DROPOUT,
    ) -> None:
        super().__init__()
        self.encoder = build_frozen_encoder(backbone, pretrained=pretrained)
        self.embed_dim = self.encoder.num_features
        self.transformer_layers = nn.ModuleList(
            [
                SupportTransformerLayer(self.embed_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(self.embed_dim)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def forward_from_embeddings(
        self,
        support_embeddings: torch.Tensor,
        query_embeddings: torch.Tensor,
        n_way: int,
        n_shot: int,
    ) -> torch.Tensor:
        support = support_embeddings.unsqueeze(0)
        for layer in self.transformer_layers:
            support = layer(support)
        support = self.norm(support).squeeze(0)
        prototypes = support.view(n_way, n_shot, -1).mean(dim=1)
        return -torch.cdist(query_embeddings, prototypes)

    def forward(self, support_images, query_images, n_way, n_shot):
        support = self.encode(support_images)
        query = self.encode(query_images)
        return self.forward_from_embeddings(support, query, n_way, n_shot)
