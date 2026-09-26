"""Create deterministic class-disjoint train/val/test splits from a class-folder dataset.

Useful for the source-dataset sensitivity experiment (e.g. CUB as an alternative source).
The default 64/16/20 proportions mirror miniImageNet; class counts are rounded while
ensuring each split is non-empty.  Symlinks avoid duplicating image data when supported.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import shutil


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--train-frac", type=float, default=0.64)
    parser.add_argument("--val-frac", type=float, default=0.16)
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    classes = sorted([p for p in args.source.iterdir() if p.is_dir()])
    random.Random(args.seed).shuffle(classes)
    if len(classes) < 15:
        raise ValueError(
            "Need at least 15 classes for useful 5-way train/val/test source splits."
        )
    n = len(classes)
    n_train = max(5, int(round(n * args.train_frac)))
    n_val = max(5, int(round(n * args.val_frac)))
    if n_train + n_val > n - 5:
        n_val = max(5, n - 5 - n_train)
    n_test = n - n_train - n_val
    if n_test < 5:
        raise ValueError("Split leaves fewer than five test classes.")
    split_map = {
        "train": classes[:n_train],
        "val": classes[n_train : n_train + n_val],
        "test": classes[n_train + n_val :],
    }

    if (
        args.output.resolve() == args.source.resolve()
        or args.output.resolve() in args.source.resolve().parents
    ):
        raise ValueError(
            "Output must not be the source directory or one of its parents."
        )
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.output} already exists; use --overwrite to replace it."
        )
    if args.output.exists() and args.overwrite:
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    for split, items in split_map.items():
        split_dir = args.output / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for source_dir in items:
            destination = split_dir / source_dir.name
            if destination.exists() or destination.is_symlink():
                continue
            if args.mode == "copy":
                shutil.copytree(source_dir, destination)
            else:
                try:
                    destination.symlink_to(
                        source_dir.resolve(), target_is_directory=True
                    )
                except OSError as exc:
                    raise OSError(
                        f"Could not create symlink {destination}. On Windows, enable Developer Mode "
                        "or rerun with --mode copy."
                    ) from exc
        print(f"{split}: {len(items)} classes")
    manifest_lines = [f"seed: {args.seed}"]
    for split, items in split_map.items():
        manifest_lines.append(f"{split}: {len(items)} classes")
        manifest_lines.extend([f"  {item.name}" for item in items])
    (args.output / "SPLIT.txt").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
