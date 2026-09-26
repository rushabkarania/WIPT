"""Train the two baselines with learned episodic heads."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from configs.experiment import BACKBONE, DEFAULT_DATA_PATHS, N_SHOT, NUM_WORKERS
from models.baselines import SupportTransformerViT, RelationHeadViT
from train.training_utils import fit, make_train_val_loaders
from utils.metrics import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["relationhead", "support_transformer"])
    parser.add_argument("--train-dir", default=DEFAULT_DATA_PATHS["miniImageNet_train"])
    parser.add_argument("--val-dir", default=DEFAULT_DATA_PATHS["miniImageNet_val"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-shot", type=int, default=N_SHOT, choices=[1, 5])
    parser.add_argument("--history", type=Path)
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = make_train_val_loaders(
        args.train_dir, args.val_dir, args.workers, n_shot=args.n_shot, seed=args.seed
    )
    if args.model == "relationhead":
        model = RelationHeadViT(backbone=BACKBONE, pretrained=True).to(device)
        output = args.output or Path("checkpoints/relation_head.pth")
        mse = nn.MSELoss()

        def loss_fn(scores, labels):
            targets = torch.zeros_like(scores)
            targets.scatter_(1, labels.unsqueeze(1), 1.0)
            return mse(scores, targets)

    else:
        model = SupportTransformerViT(
            backbone=BACKBONE, pretrained=True, num_layers=2
        ).to(device)
        output = args.output or Path("checkpoints/support_transformer.pth")
        loss_fn = nn.CrossEntropyLoss()

    print(
        f"device={device} | model={args.model} | shot={args.n_shot} | seed={args.seed}"
    )
    fit(
        model,
        train_loader,
        val_loader,
        output,
        device=device,
        checkpoint_metadata={
            "backbone": BACKBONE,
            "training_seed": args.seed,
            "n_shot": args.n_shot,
        },
        loss_fn=loss_fn,
        n_shot=args.n_shot,
        history_path=args.history,
    )


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
