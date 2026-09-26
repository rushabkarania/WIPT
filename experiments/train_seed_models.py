"""Train WIPT and the support-only control across independent training seeds."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd

from configs.experiment import DEFAULT_DATA_PATHS, TRAIN_SEEDS, NUM_WORKERS
from experiments.seed_study_common import (
    parse_seeds,
    seed_checkpoint_dir,
    seed_result_dir,
)
from utils.checkpoints import load_checkpoint


def run_command(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot", type=int, default=5, choices=[1, 5])
    parser.add_argument("--seeds", type=int, nargs="*", default=list(TRAIN_SEEDS))
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--train-dir", default=DEFAULT_DATA_PATHS["miniImageNet_train"])
    parser.add_argument("--val-dir", default=DEFAULT_DATA_PATHS["miniImageNet_val"])
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("checkpoints/seed_study"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("results/raw/training_seed_study"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["wipt", "support"],
        default=["wipt", "support"],
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    for seed in seeds:
        output_dir = seed_checkpoint_dir(args.checkpoint_root, args.shot, seed)
        output_dir.mkdir(parents=True, exist_ok=True)

        jobs = []
        if "wipt" in args.models:
            jobs.append(
                (
                    output_dir / "wipt_l2.pth",
                    [
                        sys.executable,
                        "-m",
                        "train.train_wipt",
                        "--layers",
                        "2",
                        "--seed",
                        str(seed),
                        "--n-shot",
                        str(args.shot),
                        "--workers",
                        str(args.workers),
                        "--train-dir",
                        str(args.train_dir),
                        "--val-dir",
                        str(args.val_dir),
                        "--history",
                        str(output_dir / "wipt_l2_history.csv"),
                        "--output",
                        str(output_dir / "wipt_l2.pth"),
                    ],
                )
            )
        if "support" in args.models:
            jobs.append(
                (
                    output_dir / "support_transformer.pth",
                    [
                        sys.executable,
                        "-m",
                        "train.train_baselines",
                        "support_transformer",
                        "--seed",
                        str(seed),
                        "--n-shot",
                        str(args.shot),
                        "--workers",
                        str(args.workers),
                        "--train-dir",
                        str(args.train_dir),
                        "--val-dir",
                        str(args.val_dir),
                        "--history",
                        str(output_dir / "support_transformer_history.csv"),
                        "--output",
                        str(output_dir / "support_transformer.pth"),
                    ],
                )
            )

        for output, command in jobs:
            if output.exists() and not args.overwrite:
                print(f"skip existing {output}")
                continue
            run_command(command)

    summary_rows = []
    for seed in seeds:
        directory = seed_checkpoint_dir(args.checkpoint_root, args.shot, seed)
        for model_name, filename in [
            ("WIPT", "wipt_l2.pth"),
            ("Support-only Transformer", "support_transformer.pth"),
        ]:
            path = directory / filename
            if not path.exists():
                continue
            checkpoint = load_checkpoint(path, device="cpu")
            summary_rows.append(
                {
                    "model": model_name,
                    "training_seed": seed,
                    "n_shot": args.shot,
                    "best_epoch": int(checkpoint["epoch"]),
                    "best_validation_accuracy": 100.0
                    * float(checkpoint["best_val_acc"]),
                    "validation_episode_ci95": 100.0 * float(checkpoint["val_ci"]),
                    "elapsed_seconds_to_best": checkpoint.get(
                        "elapsed_seconds_to_best"
                    ),
                    "checkpoint": str(path),
                }
            )
    output = seed_result_dir(args.out_root, args.shot)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(output / "training_checkpoints.csv", index=False)
    print("independent training-seed checkpoints complete")


if __name__ == "__main__":
    main()
