"""Train WIPT with query-group sizes q=2..5 across independent seeds.

The existing seed-study q=1 checkpoints remain the reference.  This script deliberately
writes q>1 models to a separate tree so the original results cannot be overwritten.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd

from configs.experiment import DEFAULT_DATA_PATHS, TRAIN_SEEDS, NUM_WORKERS
from experiments.seed_study_common import parse_seeds, shot_tag
from utils.checkpoints import load_checkpoint


def run(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot", type=int, default=5, choices=[1, 5])
    parser.add_argument("--query-groups", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--seeds", type=int, nargs="*", default=list(TRAIN_SEEDS))
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--train-dir", default=DEFAULT_DATA_PATHS["miniImageNet_train"])
    parser.add_argument("--val-dir", default=DEFAULT_DATA_PATHS["miniImageNet_val"])
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("checkpoints/multiquery")
    )
    parser.add_argument("--out-root", type=Path, default=Path("results/raw/multiquery"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    groups = tuple(sorted(set(args.query_groups)))
    if any(q not in (2, 3, 4, 5) for q in groups):
        raise ValueError(
            "This extension is defined for q in {2,3,4,5}; q=1 uses the existing seed study."
        )

    rows = []
    for q in groups:
        for seed in seeds:
            directory = (
                args.checkpoint_root / shot_tag(args.shot) / f"q{q}" / f"seed_{seed}"
            )
            directory.mkdir(parents=True, exist_ok=True)
            checkpoint_path = directory / "wipt_l2.pth"
            history_path = directory / "history.csv"
            if checkpoint_path.exists() and not args.overwrite:
                print(f"skip existing {checkpoint_path}")
            else:
                run(
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
                        "--query-group-size",
                        str(q),
                        "--workers",
                        str(args.workers),
                        "--train-dir",
                        str(args.train_dir),
                        "--val-dir",
                        str(args.val_dir),
                        "--output",
                        str(checkpoint_path),
                        "--history",
                        str(history_path),
                    ]
                )
            if checkpoint_path.exists():
                ckpt = load_checkpoint(checkpoint_path, device="cpu")
                rows.append(
                    {
                        "n_shot": args.shot,
                        "query_group_size": q,
                        "training_seed": seed,
                        "best_epoch": int(ckpt["epoch"]),
                        "best_validation_accuracy": 100.0 * float(ckpt["best_val_acc"]),
                        "validation_episode_ci95": 100.0 * float(ckpt["val_ci"]),
                        "elapsed_seconds_to_best": ckpt.get("elapsed_seconds_to_best"),
                        "checkpoint": str(checkpoint_path),
                        "history": str(history_path),
                    }
                )

    out = args.out_root / shot_tag(args.shot)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "training_checkpoints.csv", index=False)
    print(f"saved multi-query training metadata to {out}")


if __name__ == "__main__":
    main()
