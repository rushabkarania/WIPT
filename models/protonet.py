"""Frozen-backbone ProtoNet used in the controlled comparison."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from configs.experiment import BACKBONE
from models.backbone import build_frozen_encoder


class ProtoNet(nn.Module):
    """Class-mean ProtoNet with a permanently frozen ViT encoder."""

    def __init__(
        self,
        *,
        backbone: str = BACKBONE,
        encoder_state_dict: Mapping[str, torch.Tensor],
    ) -> None:
        super().__init__()
        self.encoder = build_frozen_encoder(
            backbone,
            pretrained=False,
            state_dict=encoder_state_dict,
        )
        self.embedding_dim = self.encoder.num_features

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if trainable != 0:
            raise RuntimeError(
                f"ProtoNet must have zero trainable parameters, found {trainable}."
            )

    def train(self, mode: bool = True):
        super().train(False)
        self.encoder.eval()
        return self

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    @staticmethod
    def compute_prototypes(
        support_embeddings: torch.Tensor,
        n_way: int,
        n_shot: int,
    ) -> torch.Tensor:
        return support_embeddings.view(n_way, n_shot, -1).mean(dim=1)

    def forward_from_embeddings(
        self,
        support_embeddings: torch.Tensor,
        query_embeddings: torch.Tensor,
        n_way: int,
        n_shot: int,
    ) -> torch.Tensor:
        prototypes = self.compute_prototypes(support_embeddings, n_way, n_shot)
        return -torch.cdist(query_embeddings, prototypes)

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
