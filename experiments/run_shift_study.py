"""Train shift-aware WIPT/support-only models over independent seeds."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd

from configs.experiment import DEFAULT_DATA_PATHS, TRAIN_SEEDS, NUM_WORKERS
from experiments.seed_study_common import parse_seeds, shot_tag
from utils.checkpoints import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot", type=int, default=5, choices=[1, 5])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["shared", "cross", "mixed"],
        default=["shared", "cross", "mixed"],
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=list(TRAIN_SEEDS))
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--shift-weight", type=float, default=0.5)
    parser.add_argument("--max-strength", type=float, default=1.0)
    parser.add_argument("--train-dir", default=DEFAULT_DATA_PATHS["miniImageNet_train"])
    parser.add_argument("--val-dir", default=DEFAULT_DATA_PATHS["miniImageNet_val"])
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("checkpoints/shift_study")
    )
    parser.add_argument(
        "--out-root", type=Path, default=Path("results/raw/shift_study")
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    summary = []
    for mode in args.modes:
        for seed in seeds:
            directory = (
                args.checkpoint_root / mode / shot_tag(args.shot) / f"seed_{seed}"
            )
            directory.mkdir(parents=True, exist_ok=True)
            for family, filename in [
                ("wipt", "wipt_l2.pth"),
                ("support", "support_transformer.pth"),
            ]:
                path = directory / filename
                history = directory / (path.stem + "_history.csv")
                if path.exists() and not args.overwrite:
                    print(f"skip existing {path}")
                else:
                    command = [
                        sys.executable,
                        "-m",
                        "train.train_shift_aware",
                        family,
                        "--mode",
                        mode,
                        "--shift-weight",
                        str(args.shift_weight),
                        "--max-strength",
                        str(args.max_strength),
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
                        "--output",
                        str(path),
                        "--history",
                        str(history),
                    ]
                    print(" ".join(command))
                    subprocess.run(command, check=True)
                if path.exists():
                    ckpt = load_checkpoint(path, device="cpu")
                    summary.append(
                        {
                            "mode": mode,
                            "model": family,
                            "n_shot": args.shot,
                            "training_seed": seed,
                            "best_epoch": int(ckpt["epoch"]),
                            "best_validation_accuracy": 100
                            * float(ckpt["best_val_acc"]),
                            "elapsed_seconds_to_best": ckpt.get(
                                "elapsed_seconds_to_best"
                            ),
                            "checkpoint": str(path),
                        }
                    )

    out = args.out_root / shot_tag(args.shot)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(out / "training_checkpoints.csv", index=False)
    print(f"saved shift-aware training metadata to {out}")


if __name__ == "__main__":
    main()
