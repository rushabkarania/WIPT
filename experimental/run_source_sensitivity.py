"""Alternative-source sensitivity study for the core WIPT/support-only comparison.

Typical use: split CUB into source train/val/test classes with
``experimental.split_source_dataset`` and train the same frozen-ViT episodic heads on
that source.  Evaluation should use target domains not used to construct the source.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from configs.experiment import NUM_WORKERS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-name", required=True)
    p.add_argument("--train-dir", required=True)
    p.add_argument("--val-dir", required=True)
    p.add_argument("--targets", nargs="+", required=True, help="NAME=PATH pairs")
    p.add_argument("--shots", type=int, nargs="+", default=[1, 5])
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    p.add_argument("--workers", type=int, default=NUM_WORKERS)
    p.add_argument(
        "--checkpoint-root", type=Path, default=Path("checkpoints/source_study")
    )
    p.add_argument("--out-root", type=Path, default=Path("results/raw/source_study"))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    source_slug = args.source_name.replace(" ", "_").lower()
    checkpoint_root = args.checkpoint_root / source_slug
    out_root = args.out_root / source_slug
    for shot in args.shots:
        train_cmd = [
            sys.executable,
            "-m",
            "experiments.train_seed_models",
            "--shot",
            str(shot),
            "--seeds",
            *map(str, args.seeds),
            "--workers",
            str(args.workers),
            "--train-dir",
            str(args.train_dir),
            "--val-dir",
            str(args.val_dir),
            "--checkpoint-root",
            str(checkpoint_root),
            "--out-root",
            str(out_root),
        ]
        if args.overwrite:
            train_cmd.append("--overwrite")
        print(" ".join(train_cmd))
        subprocess.run(train_cmd, check=True)

        eval_cmd = [
            sys.executable,
            "-m",
            "experiments.evaluate_seed_models",
            "--shot",
            str(shot),
            "--seeds",
            *map(str, args.seeds),
            "--checkpoint-root",
            str(checkpoint_root),
            "--out-root",
            str(out_root),
            "--skip-default-targets",
        ]
        for target in args.targets:
            eval_cmd.extend(["--dataset", target])
        print(" ".join(eval_cmd))
        subprocess.run(eval_cmd, check=True)


if __name__ == "__main__":
    main()
