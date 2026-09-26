"""Train WIPT with the additional margin objective used in the ablation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.experiment import BACKBONE, DEFAULT_DATA_PATHS, NUM_WORKERS
from models.wipt import WIPT
from train.training_utils import fit, make_train_val_loaders

MARGIN_LAMBDA = 0.5


def margin_term(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(scores.shape[0], device=scores.device)
    correct = scores[rows, labels]
    wrong = scores.clone()
    wrong[rows, labels] = float("-inf")
    best_wrong = wrong.max(dim=1).values
    return F.softplus(-(correct - best_wrong)).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default=DEFAULT_DATA_PATHS["miniImageNet_train"])
    parser.add_argument("--val-dir", default=DEFAULT_DATA_PATHS["miniImageNet_val"])
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/wipt_margin.pth")
    )
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = make_train_val_loaders(
        args.train_dir, args.val_dir, args.workers
    )
    model = WIPT(backbone=BACKBONE, pretrained=True, num_layers=2).to(device)
    cross_entropy = nn.CrossEntropyLoss()

    def loss_fn(scores, labels):
        return cross_entropy(scores, labels) + MARGIN_LAMBDA * margin_term(
            scores, labels
        )

    print(f"device={device} | margin_lambda={MARGIN_LAMBDA}")
    fit(
        model,
        train_loader,
        val_loader,
        args.output,
        device=device,
        checkpoint_metadata={
            "num_layers": 2,
            "backbone": BACKBONE,
            "margin_lambda": MARGIN_LAMBDA,
        },
        loss_fn=loss_fn,
    )


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
