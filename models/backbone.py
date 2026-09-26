"""Frozen ViT backbone helpers."""

from __future__ import annotations

from collections.abc import Mapping

import timm
import torch
import torch.nn as nn

from configs.experiment import BACKBONE


def build_frozen_encoder(
    backbone: str = BACKBONE,
    *,
    pretrained: bool = True,
    state_dict: Mapping[str, torch.Tensor] | None = None,
) -> nn.Module:
    """Create a frozen feature encoder, optionally from explicit weights."""
    encoder = timm.create_model(
        backbone,
        pretrained=pretrained and state_dict is None,
        num_classes=0,
    )
    if state_dict is not None:
        incompatible = encoder.load_state_dict(dict(state_dict), strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Encoder state did not load strictly: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    encoder.eval()
    return encoder
