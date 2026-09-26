"""Validate the structure of completed independent-training-seed outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from configs.experiment import TRAIN_SEEDS
from experiments.seed_study_common import parse_seeds, seed_result_dir


def require_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{path} is empty.")
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return frame


def validate_clean(root: Path, seeds: tuple[int, ...]) -> None:
    run_frame = require_csv(
        root / "clean" / "run_accuracy.csv",
        {"dataset", "model", "training_seed", "accuracy", "n_episodes"},
    )
    expected_models = {"ProtoNet", "WIPT", "Support-only Transformer"}
    if set(run_frame.model) != expected_models:
        raise ValueError("clean run_accuracy.csv has an unexpected model set.")
    for model in ("WIPT", "Support-only Transformer"):
        observed = set(
            pd.to_numeric(
                run_frame[run_frame.model == model].training_seed,
                errors="raise",
            ).astype(int)
        )
        if observed != set(seeds):
            raise ValueError(
                f"{model}: training seeds {observed}, expected {set(seeds)}"
            )

    require_csv(
        root / "clean" / "difference_summary.csv",
        {
            "dataset",
            "comparison",
            "mean_difference_pp",
            "training_seed_sd_pp",
            "n_training_seeds",
        },
    )


def validate_boundary(root: Path, seeds: tuple[int, ...]) -> None:
    transitions = require_csv(
        root / "boundary" / "transitions_by_training_seed.csv",
        {
            "training_seed",
            "n_queries",
            "rescued",
            "broken",
            "rescue_rate_percent",
            "break_rate_percent",
        },
    )
    if set(transitions.training_seed.astype(int)) != set(seeds):
        raise ValueError("boundary transition seeds do not match the requested seeds.")
    bins = require_csv(
        root / "boundary" / "margin_bins_by_training_seed.csv",
        {"training_seed", "margin_bin", "wipt_minus_protonet_pp"},
    )
    counts = bins.groupby("training_seed").margin_bin.nunique()
    if not (counts == counts.iloc[0]).all():
        raise ValueError("boundary runs do not contain the same number of margin bins.")


def validate_corruption(root: Path, seeds: tuple[int, ...]) -> None:
    gaps = require_csv(
        root / "corruption" / "corruption_gap_by_training_seed.csv",
        {"category", "severity", "training_seed", "comparison", "difference_pp"},
    )
    if set(gaps.training_seed.astype(int)) != set(seeds):
        raise ValueError("corruption seeds do not match the requested seeds.")
    require_csv(
        root / "corruption" / "corruption_gap_summary.csv",
        {
            "category",
            "severity",
            "comparison",
            "mean_difference_pp",
            "training_seed_sd_pp",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot", type=int, default=5, choices=[1, 5])
    parser.add_argument("--seeds", type=int, nargs="*", default=list(TRAIN_SEEDS))
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("results/raw/training_seed_study"),
    )
    parser.add_argument("--require-boundary", action="store_true")
    parser.add_argument("--require-corruption", action="store_true")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    root = seed_result_dir(args.out_root, args.shot)
    require_csv(
        root / "training_checkpoints.csv",
        {"model", "training_seed", "best_epoch", "best_validation_accuracy"},
    )
    validate_clean(root, seeds)
    if args.require_boundary:
        validate_boundary(root, seeds)
    if args.require_corruption:
        validate_corruption(root, seeds)
    print(f"seed-study validation passed for {root}")


if __name__ == "__main__":
    main()
