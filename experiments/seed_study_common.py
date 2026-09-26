"""Shared helpers for independent-training-seed robustness experiments."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from scipy import stats
import torch
import torch.nn as nn

from configs.experiment import BACKBONE, TRAIN_SEEDS, WIPT_DROPOUT, WIPT_HEADS
from models.backbone import build_frozen_encoder
from models.baselines import SupportTransformerViT
from models.wipt import WIPT
from utils.checkpoints import (
    encoder_state_from_checkpoint,
    load_checkpoint,
    load_wipt_state_compat,
    model_state,
    verify_encoder,
)


def parse_seeds(values: Iterable[int] | None) -> tuple[int, ...]:
    seeds = tuple(TRAIN_SEEDS if values is None else values)
    if not seeds:
        raise ValueError("At least one training seed is required.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Training seeds must be unique.")
    return seeds


def shot_tag(n_shot: int) -> str:
    if n_shot <= 0:
        raise ValueError("n_shot must be positive.")
    return f"{n_shot}shot"


def seed_checkpoint_dir(root: str | Path, n_shot: int, training_seed: int) -> Path:
    return Path(root) / shot_tag(n_shot) / f"seed_{training_seed}"


def seed_result_dir(root: str | Path, n_shot: int) -> Path:
    return Path(root) / shot_tag(n_shot)


def t_ci95(values: Iterable[float]) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(array.mean())
    if array.size == 1:
        return mean, float("nan"), float("nan")
    sd = float(array.std(ddof=1))
    half_width = float(stats.t.ppf(0.975, array.size - 1) * sd / np.sqrt(array.size))
    return mean, sd, half_width


def encode_in_chunks(
    encoder: nn.Module,
    images: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> torch.Tensor:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    parts = []
    with torch.no_grad():
        for start in range(0, images.shape[0], chunk_size):
            parts.append(encoder(images[start : start + chunk_size]))
    return torch.cat(parts, dim=0)


def protonet_scores(
    support_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    n_way: int,
    n_shot: int,
) -> torch.Tensor:
    prototypes = support_embeddings.view(n_way, n_shot, -1).mean(dim=1)
    return -torch.cdist(query_embeddings, prototypes)


def accuracy_percent(scores: torch.Tensor, labels: torch.Tensor) -> float:
    return 100.0 * float((scores.argmax(dim=1) == labels).float().mean().item())


def query_margin(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(scores.shape[0], device=scores.device)
    correct = scores[rows, labels]
    wrong = scores.clone()
    wrong[rows, labels] = float("-inf")
    return correct - wrong.max(dim=1).values


def _check_metadata(checkpoint, *, training_seed: int, n_shot: int, path: Path) -> None:
    stored_seed = checkpoint.get("training_seed")
    stored_shot = checkpoint.get("n_shot")
    if stored_seed is not None and int(stored_seed) != training_seed:
        raise RuntimeError(
            f"{path}: checkpoint training_seed={stored_seed}, expected {training_seed}."
        )
    if stored_shot is not None and int(stored_shot) != n_shot:
        raise RuntimeError(
            f"{path}: checkpoint n_shot={stored_shot}, expected {n_shot}."
        )


def _load_support_state(model: nn.Module, checkpoint) -> None:
    state = dict(model_state(checkpoint))
    state.pop("cls_token", None)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Support-only checkpoint mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )


def load_seed_study_heads(
    checkpoint_root: str | Path,
    *,
    n_shot: int,
    training_seeds: Iterable[int],
    device: torch.device,
):
    """Load one shared frozen encoder plus all independently trained episodic heads."""
    seeds = parse_seeds(training_seeds)
    first_dir = seed_checkpoint_dir(checkpoint_root, n_shot, seeds[0])
    first_wipt_path = first_dir / "wipt_l2.pth"
    first_checkpoint = load_checkpoint(first_wipt_path, device="cpu")
    _check_metadata(
        first_checkpoint,
        training_seed=seeds[0],
        n_shot=n_shot,
        path=first_wipt_path,
    )
    reference_state = encoder_state_from_checkpoint(first_checkpoint)
    encoder = (
        build_frozen_encoder(
            BACKBONE,
            pretrained=False,
            state_dict=reference_state,
        )
        .to(device)
        .eval()
    )

    wipt_heads: dict[int, WIPT] = {}
    support_heads: dict[int, SupportTransformerViT] = {}

    for seed in seeds:
        directory = seed_checkpoint_dir(checkpoint_root, n_shot, seed)
        wipt_path = directory / "wipt_l2.pth"
        support_path = directory / "support_transformer.pth"
        if not wipt_path.exists() or not support_path.exists():
            missing = [
                str(path) for path in (wipt_path, support_path) if not path.exists()
            ]
            raise FileNotFoundError(
                "Missing seed-study checkpoint(s): " + ", ".join(missing)
            )

        wipt_checkpoint = load_checkpoint(wipt_path, device="cpu")
        _check_metadata(
            wipt_checkpoint, training_seed=seed, n_shot=n_shot, path=wipt_path
        )
        wipt = WIPT(
            backbone=BACKBONE,
            pretrained=False,
            num_layers=int(wipt_checkpoint.get("num_layers", 2)),
            num_heads=WIPT_HEADS,
            dropout=WIPT_DROPOUT,
        )
        load_wipt_state_compat(wipt, wipt_checkpoint)
        verify_encoder(wipt, reference_state)
        wipt.encoder = nn.Identity()
        wipt_heads[seed] = wipt.to(device).eval()

        support_checkpoint = load_checkpoint(support_path, device="cpu")
        _check_metadata(
            support_checkpoint,
            training_seed=seed,
            n_shot=n_shot,
            path=support_path,
        )
        support = SupportTransformerViT(
            backbone=BACKBONE,
            pretrained=False,
            num_layers=2,
            num_heads=WIPT_HEADS,
            dropout=WIPT_DROPOUT,
        )
        _load_support_state(support, support_checkpoint)
        verify_encoder(support, reference_state)
        support.encoder = nn.Identity()
        support_heads[seed] = support.to(device).eval()

    return encoder, wipt_heads, support_heads
