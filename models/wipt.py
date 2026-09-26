"""Within-Instance Prototypical Transformer (WIPT)."""

from __future__ import annotations

import torch
import torch.nn as nn

from configs.experiment import BACKBONE, WIPT_DROPOUT, WIPT_HEADS, WIPT_LAYERS
from models.backbone import build_frozen_encoder


class WIPTLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = WIPT_HEADS,
        mlp_ratio: float = 4.0,
        dropout: float = WIPT_DROPOUT,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, return_attention: bool = False):
        residual = tokens
        normalized = self.norm1(tokens)
        attended, attention = self.attn(
            normalized,
            normalized,
            normalized,
            need_weights=return_attention,
            average_attn_weights=True,
        )
        tokens = residual + attended
        tokens = tokens + self.mlp(self.norm2(tokens))
        return (tokens, attention) if return_attention else tokens


class WIPT(nn.Module):
    """Jointly transform one query embedding and the episode support embeddings."""

    def __init__(
        self,
        *,
        backbone: str = BACKBONE,
        pretrained: bool = True,
        num_layers: int = WIPT_LAYERS,
        num_heads: int = WIPT_HEADS,
        mlp_ratio: float = 4.0,
        dropout: float = WIPT_DROPOUT,
        query_group_size: int = 1,
    ) -> None:
        super().__init__()
        if query_group_size <= 0:
            raise ValueError("query_group_size must be positive.")
        self.query_group_size = int(query_group_size)
        self.encoder = build_frozen_encoder(backbone, pretrained=pretrained)
        self.embed_dim = self.encoder.num_features
        self.transformer_layers = nn.ModuleList(
            [
                WIPTLayer(
                    embed_dim=self.embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(self.embed_dim)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def _apply_transformer(
        self, tokens: torch.Tensor, *, return_attention: bool = False
    ):
        attention = None
        for index, layer in enumerate(self.transformer_layers):
            if return_attention and index == len(self.transformer_layers) - 1:
                tokens, attention = layer(tokens, return_attention=True)
            else:
                tokens = layer(tokens)
        return self.norm(tokens), attention

    def transform_embeddings(
        self,
        support_embeddings: torch.Tensor,
        query_embeddings: torch.Tensor,
        n_way: int,
        n_shot: int,
        *,
        return_attention: bool = False,
        query_group_size: int | None = None,
    ):
        """Transform embeddings with one or more query tokens per sequence.

        ``query_group_size=1`` is exactly the original WIPT computation: every
        query is processed independently with its own copy of the support set.
        For ``query_group_size>1``, consecutive unlabeled queries are grouped in
        the same Transformer sequence and therefore share a query-conditioned
        prototype set.  No query labels are used to form the groups.  The final
        incomplete group, if any, is processed at its natural smaller size so
        that every evaluation query is retained.
        """
        group_size = (
            self.query_group_size if query_group_size is None else int(query_group_size)
        )
        if group_size <= 0:
            raise ValueError("query_group_size must be positive.")
        n_query = int(query_embeddings.shape[0])
        if n_query == 0:
            raise ValueError("At least one query embedding is required.")

        # Preserve the original vectorized path exactly for checkpoint-compatible q=1 WIPT.
        if group_size == 1:
            support_tokens = support_embeddings.unsqueeze(0).expand(n_query, -1, -1)
            tokens = torch.cat([query_embeddings.unsqueeze(1), support_tokens], dim=1)
            tokens, attention = self._apply_transformer(
                tokens, return_attention=return_attention
            )
            transformed_query = tokens[:, 0]
            transformed_support = tokens[:, 1:].view(n_query, n_way, n_shot, -1)
            prototypes = transformed_support.mean(dim=2)
            return transformed_query, transformed_support, prototypes, attention

        if return_attention:
            raise NotImplementedError(
                "Attention export is currently defined only for single-query WIPT; "
                "multi-query groups have variable sequence membership."
            )

        # Vectorize all complete query groups in one Transformer batch.  This is
        # materially faster than a Python loop and gives the multi-query condition a
        # fair latency comparison.  A possible final incomplete group is handled once.
        transformed_query_parts = []
        transformed_support_parts = []
        prototype_parts = []
        n_full_groups = n_query // group_size
        n_full_queries = n_full_groups * group_size

        if n_full_groups:
            query_groups = query_embeddings[:n_full_queries].reshape(
                n_full_groups, group_size, -1
            )
            support_tokens = support_embeddings.unsqueeze(0).expand(
                n_full_groups, -1, -1
            )
            tokens = torch.cat([query_groups, support_tokens], dim=1)
            tokens, _ = self._apply_transformer(tokens, return_attention=False)
            query_out = tokens[:, :group_size].reshape(n_full_queries, -1)
            support_out = tokens[:, group_size:].view(n_full_groups, n_way, n_shot, -1)
            group_prototypes = support_out.mean(dim=2)
            transformed_query_parts.append(query_out)
            transformed_support_parts.append(
                support_out.unsqueeze(1)
                .expand(-1, group_size, -1, -1, -1)
                .reshape(n_full_queries, n_way, n_shot, -1)
            )
            prototype_parts.append(
                group_prototypes.unsqueeze(1)
                .expand(-1, group_size, -1, -1)
                .reshape(n_full_queries, n_way, -1)
            )

        if n_full_queries < n_query:
            remainder = query_embeddings[n_full_queries:]
            current = int(remainder.shape[0])
            tokens = torch.cat(
                [remainder.unsqueeze(0), support_embeddings.unsqueeze(0)], dim=1
            )
            tokens, _ = self._apply_transformer(tokens, return_attention=False)
            query_out = tokens[0, :current]
            support_out = tokens[0, current:].view(n_way, n_shot, -1)
            group_prototypes = support_out.mean(dim=1)
            transformed_query_parts.append(query_out)
            transformed_support_parts.append(
                support_out.unsqueeze(0).expand(current, -1, -1, -1)
            )
            prototype_parts.append(
                group_prototypes.unsqueeze(0).expand(current, -1, -1)
            )

        transformed_query = torch.cat(transformed_query_parts, dim=0)
        transformed_support = torch.cat(transformed_support_parts, dim=0)
        prototypes = torch.cat(prototype_parts, dim=0)
        return transformed_query, transformed_support, prototypes, None

    def forward_from_embeddings(
        self,
        support_embeddings: torch.Tensor,
        query_embeddings: torch.Tensor,
        n_way: int,
        n_shot: int,
        *,
        return_attention: bool = False,
    ):
        transformed_query, transformed_support, prototypes, attention = (
            self.transform_embeddings(
                support_embeddings,
                query_embeddings,
                n_way,
                n_shot,
                return_attention=return_attention,
                query_group_size=self.query_group_size,
            )
        )
        scores = -torch.cdist(transformed_query.unsqueeze(1), prototypes).squeeze(1)
        if return_attention:
            return scores, transformed_query, transformed_support, prototypes, attention
        return scores, transformed_query, transformed_support, prototypes

    def forward(
        self,
        support_images: torch.Tensor,
        query_images: torch.Tensor,
        n_way: int,
        n_shot: int,
    ) -> torch.Tensor:
        support = self.encode(support_images)
        query = self.encode(query_images)
        scores, _, _, _ = self.forward_from_embeddings(support, query, n_way, n_shot)
        return scores

    def forward_with_attention(
        self,
        support_images: torch.Tensor,
        query_images: torch.Tensor,
        n_way: int,
        n_shot: int,
    ):
        support = self.encode(support_images)
        query = self.encode(query_images)
        scores, _, _, _, attention = self.forward_from_embeddings(
            support,
            query,
            n_way,
            n_shot,
            return_attention=True,
        )
        return scores, attention
