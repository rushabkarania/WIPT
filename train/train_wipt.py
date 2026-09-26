"""Train standard WIPT with a frozen ViT encoder."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from configs.experiment import (
    BACKBONE,
    DEFAULT_DATA_PATHS,
    N_SHOT,
    WIPT_DROPOUT,
    WIPT_HEADS,
    NUM_WORKERS,
)
from models.wipt import WIPT
from train.training_utils import fit, make_train_val_loaders
from utils.metrics import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=2, choices=[2, 4, 6])
    parser.add_argument("--train-dir", default=DEFAULT_DATA_PATHS["miniImageNet_train"])
    parser.add_argument("--val-dir", default=DEFAULT_DATA_PATHS["miniImageNet_val"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-shot", type=int, default=N_SHOT, choices=[1, 5])
    parser.add_argument(
        "--query-group-size", type=int, default=1, choices=[1, 2, 3, 4, 5]
    )
    parser.add_argument("--history", type=Path)
    args = parser.parse_args()

    output = args.output or Path(f"checkpoints/wipt_l{args.layers}.pth")
    if args.seed is not None:
        set_seed(args.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = make_train_val_loaders(
        args.train_dir,
        args.val_dir,
        args.workers,
        n_shot=args.n_shot,
        seed=args.seed,
        # Multi-query groups must not inherit the class-major query ordering of
        # the episodic collate function. q=1 deliberately preserves the legacy path.
        shuffle_queries=args.query_group_size > 1,
    )
    model = WIPT(
        backbone=BACKBONE,
        pretrained=True,
        num_layers=args.layers,
        num_heads=WIPT_HEADS,
        dropout=WIPT_DROPOUT,
        query_group_size=args.query_group_size,
    ).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"device={device} | layers={args.layers} | shot={args.n_shot} | "
        f"seed={args.seed} | query_group_size={args.query_group_size} | "
        f"trainable_parameters={trainable:,}"
    )
    fit(
        model,
        train_loader,
        val_loader,
        output,
        device=device,
        checkpoint_metadata={
            "num_layers": args.layers,
            "backbone": BACKBONE,
            "training_seed": args.seed,
            "n_shot": args.n_shot,
            "query_group_size": args.query_group_size,
        },
        n_shot=args.n_shot,
        history_path=args.history,
    )


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
