"""Evaluation metrics and reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def accuracy(scores: torch.Tensor, labels: torch.Tensor) -> float:
    return (scores.argmax(dim=1) == labels).float().mean().item()


def mean_ci95(values) -> tuple[float, float]:
    values = np.asarray(list(values), dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    mean = float(values.mean())
    if len(values) == 1:
        return mean, 0.0
    ci = float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, ci


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    ``deterministic=True`` also disables cuDNN benchmarking and requests
    deterministic cuDNN kernels where available.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
