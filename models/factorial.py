"""Factorial few-shot controls for the controlled study.

The study crosses three decision rules with three representation/adaptation modes:

    decision rule:  euclidean prototype | cosine matching | learned relation
    adaptation:     none | support-only Transformer | joint query/support Transformer

This yields the nine conditions requested in the manuscript feedback while keeping the
frozen ViT representation and episodic protocol controlled.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.experiment import BACKBONE, WIPT_DROPOUT, WIPT_HEADS, WIPT_LAYERS
from models.backbone import build_frozen_encoder
from models.baselines import SupportTransformerLayer
from models.wipt import WIPTLayer

ADAPTATION_MODES = ("none", "support", "joint")
SCORERS = ("euclidean", "cosine", "relation")


def condition_name(adaptation: str, scorer: str, query_group_size: int = 1) -> str:
    if adaptation not in ADAPTATION_MODES:
        raise ValueError(adaptation)
    if scorer not in SCORERS:
        raise ValueError(scorer)
    prefix = {"none": "Raw", "support": "Support-only", "joint": "WIPT"}[adaptation]
    suffix = {"euclidean": "Euclidean", "cosine": "Cosine", "relation": "Relation"}[
        scorer
    ]
    if adaptation == "joint" and query_group_size != 1:
        prefix = f"WIPT-q{query_group_size}"
    return f"{prefix} + {suffix}"


class RelationScorer(nn.Module):
    """Relation-Network-style score over query/prototype pairs."""

    def __init__(self, embed_dim: int, relation_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, relation_dim),
            nn.ReLU(),
            nn.Linear(relation_dim, relation_dim // 2),
            nn.ReLU(),
            nn.Linear(relation_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, query: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """Score [Q,D] queries against [C,D] or [Q,C,D] prototypes."""
        if prototypes.ndim == 2:
            proto = prototypes.unsqueeze(0).expand(query.shape[0], -1, -1)
        elif prototypes.ndim == 3:
            if prototypes.shape[0] != query.shape[0]:
                raise ValueError("Per-query prototypes must match query count.")
            proto = prototypes
        else:
            raise ValueError("prototypes must have shape [C,D] or [Q,C,D].")
        q = query.unsqueeze(1).expand(-1, proto.shape[1], -1)
        pairs = torch.cat([q, proto], dim=-1)
        return self.net(pairs.reshape(-1, pairs.shape[-1])).view(
            query.shape[0], proto.shape[1]
        )


class FactorialFewShotModel(nn.Module):
    """Frozen encoder plus a controlled adaptation mode and decision rule."""

    def __init__(
        self,
        *,
        adaptation: str,
        scorer: str,
        backbone: str = BACKBONE,
        pretrained: bool = True,
        encoder_state_dict: Mapping[str, torch.Tensor] | None = None,
        num_layers: int = WIPT_LAYERS,
        num_heads: int = WIPT_HEADS,
        dropout: float = WIPT_DROPOUT,
        mlp_ratio: float = 4.0,
        relation_dim: int = 256,
        query_group_size: int = 1,
    ) -> None:
        super().__init__()
        if adaptation not in ADAPTATION_MODES:
            raise ValueError(f"Unknown adaptation mode: {adaptation}")
        if scorer not in SCORERS:
            raise ValueError(f"Unknown scorer: {scorer}")
        if query_group_size <= 0:
            raise ValueError("query_group_size must be positive.")

        self.adaptation = adaptation
        self.scorer = scorer
        self.query_group_size = int(query_group_size)
        self.encoder = build_frozen_encoder(
            backbone,
            pretrained=pretrained and encoder_state_dict is None,
            state_dict=encoder_state_dict,
        )
        self.embed_dim = self.encoder.num_features

        if adaptation == "support":
            self.transformer_layers = nn.ModuleList(
                [
                    SupportTransformerLayer(
                        self.embed_dim,
                        num_heads,
                        dropout,
                        mlp_ratio=mlp_ratio,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.norm = nn.LayerNorm(self.embed_dim)
        elif adaptation == "joint":
            self.transformer_layers = nn.ModuleList(
                [
                    WIPTLayer(
                        embed_dim=self.embed_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        mlp_ratio=mlp_ratio,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.norm = nn.LayerNorm(self.embed_dim)
        else:
            self.transformer_layers = nn.ModuleList()
            self.norm = nn.Identity()

        self.relation = (
            RelationScorer(self.embed_dim, relation_dim)
            if scorer == "relation"
            else None
        )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def _support_transform(self, support: torch.Tensor) -> torch.Tensor:
        tokens = support.unsqueeze(0)
        for layer in self.transformer_layers:
            tokens = layer(tokens)
        return self.norm(tokens).squeeze(0)

    def _joint_group_transform(
        self,
        support: torch.Tensor,
        queries: torch.Tensor,
        n_way: int,
        n_shot: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return transformed queries and per-query transformed support [Q,S,D]."""
        q_out_parts = []
        s_out_parts = []
        group_size = self.query_group_size
        for start in range(0, queries.shape[0], group_size):
            group = queries[start : start + group_size]
            current = group.shape[0]
            tokens = torch.cat([group.unsqueeze(0), support.unsqueeze(0)], dim=1)
            for layer in self.transformer_layers:
                tokens = layer(tokens)
            tokens = self.norm(tokens)
            q_out = tokens[0, :current]
            s_out = tokens[0, current:].view(n_way * n_shot, -1)
            q_out_parts.append(q_out)
            s_out_parts.append(s_out.unsqueeze(0).expand(current, -1, -1))
        return torch.cat(q_out_parts, dim=0), torch.cat(s_out_parts, dim=0)

    @staticmethod
    def _class_means(support: torch.Tensor, n_way: int, n_shot: int) -> torch.Tensor:
        if support.ndim == 2:
            return support.view(n_way, n_shot, -1).mean(dim=1)
        if support.ndim == 3:
            q = support.shape[0]
            return support.view(q, n_way, n_shot, -1).mean(dim=2)
        raise ValueError("support must be [S,D] or [Q,S,D].")

    @staticmethod
    def _euclidean_scores(
        query: torch.Tensor, prototypes: torch.Tensor
    ) -> torch.Tensor:
        if prototypes.ndim == 2:
            return -torch.cdist(query, prototypes)
        return -torch.linalg.vector_norm(query.unsqueeze(1) - prototypes, dim=-1)

    @staticmethod
    def _cosine_scores(
        query: torch.Tensor,
        support: torch.Tensor,
        n_way: int,
        n_shot: int,
    ) -> torch.Tensor:
        query = F.normalize(query, dim=-1)
        if support.ndim == 2:
            support = F.normalize(support, dim=-1)
            similarity = query @ support.T
            return similarity.view(query.shape[0], n_way, n_shot).mean(dim=2)
        support = F.normalize(support, dim=-1)
        similarity = (query.unsqueeze(1) * support).sum(dim=-1)
        return similarity.view(query.shape[0], n_way, n_shot).mean(dim=2)

    def forward_from_embeddings(
        self,
        support_embeddings: torch.Tensor,
        query_embeddings: torch.Tensor,
        n_way: int,
        n_shot: int,
    ) -> torch.Tensor:
        if self.adaptation == "none":
            support = support_embeddings
            query = query_embeddings
        elif self.adaptation == "support":
            support = self._support_transform(support_embeddings)
            query = query_embeddings
        else:
            query, support = self._joint_group_transform(
                support_embeddings,
                query_embeddings,
                n_way,
                n_shot,
            )

        if self.scorer == "euclidean":
            prototypes = self._class_means(support, n_way, n_shot)
            return self._euclidean_scores(query, prototypes)
        if self.scorer == "cosine":
            return self._cosine_scores(query, support, n_way, n_shot)
        prototypes = self._class_means(support, n_way, n_shot)
        assert self.relation is not None
        return self.relation(query, prototypes)

    def forward(
        self,
        support_images: torch.Tensor,
        query_images: torch.Tensor,
        n_way: int,
        n_shot: int,
    ) -> torch.Tensor:
        support = self.encode(support_images)
        query = self.encode(query_images)
        return self.forward_from_embeddings(support, query, n_way, n_shot)


def loss_for_scorer(scorer: str):
    if scorer == "relation":
        mse = nn.MSELoss()

        def relation_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            targets = torch.zeros_like(scores)
            targets.scatter_(1, labels.unsqueeze(1), 1.0)
            return mse(scores, targets)

        return relation_loss
    return nn.CrossEntropyLoss()
