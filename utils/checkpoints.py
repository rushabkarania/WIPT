"""Checkpoint loading and frozen-encoder verification."""

from __future__ import annotations

from pathlib import Path

import torch


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def model_state(checkpoint) -> dict[str, torch.Tensor]:
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a model state dictionary.")
    return state


def encoder_state_from_checkpoint(checkpoint) -> dict[str, torch.Tensor]:
    state = model_state(checkpoint)
    prefix = "encoder."
    encoder_state = {
        key[len(prefix) :]: value.detach().cpu()
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not encoder_state:
        raise RuntimeError("No encoder tensors were found in the checkpoint.")
    return encoder_state


def load_wipt_state_compat(model, checkpoint) -> None:
    """Load older WIPT checkpoints after removing the unused cls_token from older checkpoints."""
    state = dict(model_state(checkpoint))
    state.pop("cls_token", None)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "WIPT checkpoint mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )


def verify_encoder(model, expected_state: dict[str, torch.Tensor]) -> None:
    actual = model.encoder.state_dict()
    if set(actual) != set(expected_state):
        raise RuntimeError("Encoder keys do not match the reference WIPT encoder.")
    mismatched = [
        key
        for key in expected_state
        if not torch.equal(actual[key].detach().cpu(), expected_state[key])
    ]
    if mismatched:
        raise RuntimeError(
            f"Encoder differs from the reference checkpoint: {mismatched[:5]}"
        )
