"""Train all trainable conditions in the requested 3 x 3 factorial study."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd

from configs.experiment import DEFAULT_DATA_PATHS, TRAIN_SEEDS, NUM_WORKERS
from experiments.seed_study_common import parse_seeds, shot_tag
from models.factorial import ADAPTATION_MODES, SCORERS, condition_name
from utils.checkpoints import load_checkpoint


def slug(adaptation: str, scorer: str) -> str:
    return f"{adaptation}_{scorer}"


def requires_training(adaptation: str, scorer: str) -> bool:
    # Raw Euclidean/cosine have no trainable episodic parameters.
    # The Euclidean support-only and joint/WIPT conditions already exist in
    # checkpoints/seed_study and are reused rather than retrained.
    if adaptation == "none":
        return scorer == "relation"
    if scorer == "euclidean":
        return False
    return True


def reused_seed_checkpoint(
    reference_root: Path, shot: int, adaptation: str, seed: int
) -> Path:
    base = reference_root / shot_tag(shot) / f"seed_{seed}"
    if adaptation == "support":
        return base / "support_transformer.pth"
    if adaptation == "joint":
        return base / "wipt_l2.pth"
    raise ValueError(f"No reused checkpoint for adaptation={adaptation!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot", type=int, default=5, choices=[1, 5])
    parser.add_argument("--seeds", type=int, nargs="*", default=list(TRAIN_SEEDS))
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--train-dir", default=DEFAULT_DATA_PATHS["miniImageNet_train"])
    parser.add_argument("--val-dir", default=DEFAULT_DATA_PATHS["miniImageNet_val"])
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("checkpoints/factorial")
    )
    parser.add_argument(
        "--reference-checkpoint-root", type=Path, default=Path("checkpoints/seed_study")
    )
    parser.add_argument("--out-root", type=Path, default=Path("results/raw/factorial"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    rows = []
    for adaptation in ADAPTATION_MODES:
        for scorer in SCORERS:
            if adaptation == "none" and not requires_training(adaptation, scorer):
                rows.append(
                    {
                        "adaptation": adaptation,
                        "scorer": scorer,
                        "condition": condition_name(adaptation, scorer),
                        "n_shot": args.shot,
                        "training_seed": "fixed",
                        "best_epoch": None,
                        "best_validation_accuracy": None,
                        "elapsed_seconds_to_best": None,
                        "checkpoint": None,
                        "source": "fixed_no_training",
                    }
                )
                continue

            if adaptation in {"support", "joint"} and scorer == "euclidean":
                for seed in seeds:
                    checkpoint = reused_seed_checkpoint(
                        args.reference_checkpoint_root, args.shot, adaptation, seed
                    )
                    if not checkpoint.exists():
                        raise FileNotFoundError(
                            f"Missing existing seed-study checkpoint required for reuse: {checkpoint}"
                        )
                    ckpt = load_checkpoint(checkpoint, device="cpu")
                    rows.append(
                        {
                            "adaptation": adaptation,
                            "scorer": scorer,
                            "condition": condition_name(adaptation, scorer),
                            "n_shot": args.shot,
                            "training_seed": seed,
                            "best_epoch": int(ckpt["epoch"]),
                            "best_validation_accuracy": 100.0
                            * float(ckpt["best_val_acc"]),
                            "elapsed_seconds_to_best": ckpt.get(
                                "elapsed_seconds_to_best"
                            ),
                            "checkpoint": str(checkpoint),
                            "source": "reused_seed_study",
                        }
                    )
                print(
                    f"reuse existing {condition_name(adaptation, scorer)} seed-study checkpoints"
                )
                continue

            for seed in seeds:
                directory = (
                    args.checkpoint_root
                    / shot_tag(args.shot)
                    / slug(adaptation, scorer)
                    / f"seed_{seed}"
                )
                directory.mkdir(parents=True, exist_ok=True)
                checkpoint = directory / "model.pth"
                history = directory / "history.csv"
                if checkpoint.exists() and not args.overwrite:
                    print(f"skip existing {checkpoint}")
                else:
                    command = [
                        sys.executable,
                        "-m",
                        "train.train_factorial",
                        "--adaptation",
                        adaptation,
                        "--scorer",
                        scorer,
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
                        str(checkpoint),
                        "--history",
                        str(history),
                    ]
                    print(" ".join(command))
                    subprocess.run(command, check=True)
                if checkpoint.exists():
                    ckpt = load_checkpoint(checkpoint, device="cpu")
                    rows.append(
                        {
                            "adaptation": adaptation,
                            "scorer": scorer,
                            "condition": condition_name(adaptation, scorer),
                            "n_shot": args.shot,
                            "training_seed": seed,
                            "best_epoch": int(ckpt["epoch"]),
                            "best_validation_accuracy": 100.0
                            * float(ckpt["best_val_acc"]),
                            "elapsed_seconds_to_best": ckpt.get(
                                "elapsed_seconds_to_best"
                            ),
                            "checkpoint": str(checkpoint),
                            "source": "factorial_trained",
                        }
                    )

    out = args.out_root / shot_tag(args.shot)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "training_checkpoints.csv", index=False)
    print(f"saved factorial training metadata to {out}")


if __name__ == "__main__":
    main()
