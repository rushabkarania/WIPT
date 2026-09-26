"""Train the residual and combined WIPT refinements used in the paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from configs.experiment import BACKBONE, DEFAULT_DATA_PATHS, NUM_WORKERS
from models.wipt_variants import RefinedWIPT
from train.training_utils import fit, make_train_val_loaders

VARIANTS = {
    "residual": {
        "checkpoint": "checkpoints/wipt_residual.pth",
        "kwargs": {
            "attention_temperature": 1.0,
            "support_dropout": 0.0,
            "input_norm": False,
            "residual_query": True,
        },
    },
    "all": {
        "checkpoint": "checkpoints/wipt_all.pth",
        "kwargs": {
            "attention_temperature": 2.0,
            "support_dropout": 0.1,
            "input_norm": True,
            "residual_query": True,
        },
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("--train-dir", default=DEFAULT_DATA_PATHS["miniImageNet_train"])
    parser.add_argument("--val-dir", default=DEFAULT_DATA_PATHS["miniImageNet_val"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()

    spec = VARIANTS[args.variant]
    output = args.output or Path(spec["checkpoint"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = make_train_val_loaders(
        args.train_dir, args.val_dir, args.workers
    )
    model = RefinedWIPT(backbone=BACKBONE, pretrained=True, **spec["kwargs"]).to(device)
    print(f"device={device} | variant={args.variant}")
    fit(
        model,
        train_loader,
        val_loader,
        output,
        device=device,
        checkpoint_metadata={
            "num_layers": 2,
            "backbone": BACKBONE,
            "refinement_kwargs": spec["kwargs"],
        },
    )


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
