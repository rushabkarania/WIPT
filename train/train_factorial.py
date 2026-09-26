"""Train one condition from the 3 x 3 factorial study."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from configs.experiment import BACKBONE, DEFAULT_DATA_PATHS, N_SHOT, NUM_WORKERS
from models.factorial import (
    ADAPTATION_MODES,
    SCORERS,
    FactorialFewShotModel,
    condition_name,
    loss_for_scorer,
)
from train.training_utils import fit, make_train_val_loaders
from utils.metrics import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptation", choices=ADAPTATION_MODES, required=True)
    parser.add_argument("--scorer", choices=SCORERS, required=True)
    parser.add_argument("--train-dir", default=DEFAULT_DATA_PATHS["miniImageNet_train"])
    parser.add_argument("--val-dir", default=DEFAULT_DATA_PATHS["miniImageNet_val"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-shot", type=int, default=N_SHOT, choices=[1, 5])
    parser.add_argument(
        "--query-group-size", type=int, default=1, choices=[1, 2, 3, 4, 5]
    )
    args = parser.parse_args()

    if args.adaptation != "joint" and args.query_group_size != 1:
        raise ValueError("query_group_size only changes the joint/WIPT condition.")

    set_seed(args.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FactorialFewShotModel(
        adaptation=args.adaptation,
        scorer=args.scorer,
        backbone=BACKBONE,
        pretrained=True,
        query_group_size=args.query_group_size,
    ).to(device)
    if model.trainable_parameter_count == 0:
        raise ValueError(
            f"{condition_name(args.adaptation, args.scorer)} has no trainable episodic parameters; "
            "evaluate it directly instead of training a checkpoint."
        )

    train_loader, val_loader = make_train_val_loaders(
        args.train_dir,
        args.val_dir,
        args.workers,
        n_shot=args.n_shot,
        seed=args.seed,
    )
    print(
        f"device={device} | condition={condition_name(args.adaptation, args.scorer, args.query_group_size)} "
        f"| shot={args.n_shot} | seed={args.seed} | trainable_parameters={model.trainable_parameter_count:,}"
    )
    fit(
        model,
        train_loader,
        val_loader,
        args.output,
        device=device,
        checkpoint_metadata={
            "backbone": BACKBONE,
            "adaptation": args.adaptation,
            "scorer": args.scorer,
            "training_seed": args.seed,
            "n_shot": args.n_shot,
            "query_group_size": args.query_group_size,
            "condition": condition_name(
                args.adaptation, args.scorer, args.query_group_size
            ),
        },
        loss_fn=loss_for_scorer(args.scorer),
        n_shot=args.n_shot,
        history_path=args.history,
    )


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
