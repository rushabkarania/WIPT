"""Convenience runner for the independent-training-seed study."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from configs.experiment import TRAIN_SEEDS, NUM_WORKERS


def run(module: str, arguments: list[str]) -> None:
    command = [sys.executable, "-m", module, *arguments]
    print(" ".join(command))
    subprocess.run(command, check=True)


def common_arguments(args) -> list[str]:
    values = ["--shot", str(args.shot), "--seeds", *map(str, args.seeds)]
    values.extend(["--checkpoint-root", str(args.checkpoint_root)])
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=["train", "clean", "boundary", "corruption", "core", "full"],
        help="core=train+clean; full=train+clean+boundary+corruption",
    )
    parser.add_argument("--shot", type=int, default=5, choices=[1, 5])
    parser.add_argument("--seeds", type=int, nargs="*", default=list(TRAIN_SEEDS))
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-in-domain", action="store_true")
    args = parser.parse_args()

    stages = {
        "train": ["train"],
        "clean": ["clean"],
        "boundary": ["boundary"],
        "corruption": ["corruption"],
        "core": ["train", "clean"],
        "full": ["train", "clean", "boundary", "corruption"],
    }[args.stage]

    common = common_arguments(args)
    if "train" in stages:
        train_args = [
            *common,
            "--workers",
            str(args.workers),
            "--out-root",
            str(args.out_root),
        ]
        if args.overwrite:
            train_args.append("--overwrite")
        run("experiments.train_seed_models", train_args)

    if "clean" in stages:
        clean_args = [*common, "--out-root", str(args.out_root)]
        if args.include_in_domain:
            clean_args.append("--include-in-domain")
        run("experiments.evaluate_seed_models", clean_args)

    if "boundary" in stages:
        run(
            "experiments.analyze_boundary_across_seeds",
            [*common, "--out-root", str(args.out_root)],
        )

    if "corruption" in stages:
        run(
            "experiments.evaluate_corruption_across_seeds",
            [*common, "--out-root", str(args.out_root)],
        )


if __name__ == "__main__":
    main()
