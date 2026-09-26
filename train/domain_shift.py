"""Episode-level synthetic pseudo-domain shifts for source-only CD-FSL training.

The shift is sampled once per episode so all images in that pseudo-domain share the
same appearance parameters.  Gaussian noise remains image-specific.  Inputs/outputs
use the ImageNet-normalized tensor space expected by the frozen ViT.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from configs.experiment import IMAGENET_MEAN, IMAGENET_STD


@dataclass(frozen=True)
class ShiftParams:
    brightness: float
    contrast: float
    saturation: float
    gamma: float
    noise_sigma: float
    red_scale: float
    blue_scale: float


def _uniform(low: float, high: float) -> float:
    return float(low + (high - low) * torch.rand(()).item())


def sample_shift(strength: float = 1.0) -> ShiftParams:
    """Sample a semantics-preserving appearance shift.

    ``strength=0`` is approximately identity and ``strength=1`` spans the full
    training range.  The ranges are intentionally symmetric where appropriate so
    the model is not trained only on the positive-brightness corruption used in
    the paper's diagnostic benchmark.
    """
    strength = float(max(0.0, min(1.0, strength)))
    return ShiftParams(
        brightness=_uniform(-0.25 * strength, 0.25 * strength),
        contrast=_uniform(1.0 - 0.40 * strength, 1.0 + 0.40 * strength),
        saturation=_uniform(1.0 - 0.45 * strength, 1.0 + 0.45 * strength),
        gamma=_uniform(1.0 - 0.30 * strength, 1.0 + 0.40 * strength),
        noise_sigma=_uniform(0.0, 0.16 * strength),
        red_scale=_uniform(1.0 - 0.15 * strength, 1.0 + 0.15 * strength),
        blue_scale=_uniform(1.0 - 0.15 * strength, 1.0 + 0.15 * strength),
    )


def _stats(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.as_tensor(
        IMAGENET_MEAN, device=images.device, dtype=images.dtype
    ).view(1, 3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, device=images.device, dtype=images.dtype).view(
        1, 3, 1, 1
    )
    return mean, std


def apply_shift(images: torch.Tensor, params: ShiftParams) -> torch.Tensor:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Expected normalized image tensor [N,3,H,W].")
    mean, std = _stats(images)
    x = (images * std + mean).clamp(0.0, 1.0)

    # Global pseudo-domain appearance parameters.
    x = x + params.brightness
    per_image_mean = x.mean(dim=(2, 3), keepdim=True)
    x = per_image_mean + params.contrast * (x - per_image_mean)

    gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
    x = gray + params.saturation * (x - gray)

    channel_scale = torch.tensor(
        [params.red_scale, 1.0, params.blue_scale], device=x.device, dtype=x.dtype
    ).view(1, 3, 1, 1)
    x = x * channel_scale
    x = x.clamp(1e-5, 1.0).pow(params.gamma)

    if params.noise_sigma > 0:
        x = x + torch.randn_like(x) * params.noise_sigma
    x = x.clamp(0.0, 1.0)
    return (x - mean) / std


def shift_episode(
    support: torch.Tensor,
    query: torch.Tensor,
    *,
    mode: str,
    strength: float,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Apply one of the source-only pseudo-domain training regimes.

    shared: support and query share one pseudo-domain (closest to target-domain episodes)
    cross:  support and query receive independent pseudo-domains (adaptation stress test)
    mixed:  randomly select shared or cross per episode
    """
    if mode == "mixed":
        mode = "shared" if bool(torch.rand(()) < 0.5) else "cross"
    if mode not in {"shared", "cross"}:
        raise ValueError("mode must be shared, cross, or mixed")
    support_params = sample_shift(strength)
    query_params = support_params if mode == "shared" else sample_shift(strength)
    return apply_shift(support, support_params), apply_shift(query, query_params), mode
