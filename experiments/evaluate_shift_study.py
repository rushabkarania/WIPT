"""Evaluate each source-shift training regime with the standard seed-study evaluator."""

from __future__ import annotations
import argparse
from pathlib import Path
import subprocess, sys

from configs.experiment import DEFAULT_DATA_PATHS, TRAIN_SEEDS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shot", type=int, required=True, choices=[1, 5])
    p.add_argument(
        "--modes",
        nargs="+",
        default=["shared", "cross", "mixed"],
        choices=["shared", "cross", "mixed"],
    )
    p.add_argument("--seeds", type=int, nargs="*", default=list(TRAIN_SEEDS))
    p.add_argument(
        "--checkpoint-root", type=Path, default=Path("checkpoints/shift_study")
    )
    p.add_argument("--out-root", type=Path, default=Path("results/raw/shift_study"))
    p.add_argument("--cub", default=DEFAULT_DATA_PATHS["CUB"])
    p.add_argument("--eurosat", default=DEFAULT_DATA_PATHS["EuroSAT"])
    p.add_argument("--isic", default=DEFAULT_DATA_PATHS["ISIC"])
    p.add_argument(
        "--dataset", action="append", default=[], help="Additional NAME=PATH"
    )
    a = p.parse_args()
    for mode in a.modes:
        cmd = [
            sys.executable,
            "-m",
            "experiments.evaluate_seed_models",
            "--shot",
            str(a.shot),
            "--seeds",
            *map(str, a.seeds),
            "--checkpoint-root",
            str(a.checkpoint_root / mode),
            "--out-root",
            str(a.out_root / mode),
            "--cub",
            str(a.cub),
            "--eurosat",
            str(a.eurosat),
            "--isic",
            str(a.isic),
        ]
        for item in a.dataset:
            cmd.extend(["--dataset", item])
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
