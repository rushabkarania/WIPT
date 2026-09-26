"""Evaluate the paper models on miniImageNet test episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from configs.experiment import (
    DEFAULT_DATA_PATHS,
    EVAL_SEEDS,
    INDOMAIN_EPISODES_PER_SEED,
    N_SHOT,
    N_WAY,
)
from eval.runtime import episode_loader, get_device, load_paper_models
from utils.metrics import accuracy


def evaluate_model(model, data_path, seed, device):
    values = []
    with torch.no_grad():
        for support, _, query, labels in episode_loader(
            data_path, seed, INDOMAIN_EPISODES_PER_SEED
        ):
            support, query, labels = (
                support.to(device),
                query.to(device),
                labels.to(device),
            )
            values.append(accuracy(model(support, query, N_WAY, N_SHOT), labels))
    return 100 * float(np.mean(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA_PATHS["miniImageNet_test"])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/raw/clean/in_domain_accuracy.csv"),
    )
    args = parser.parse_args()

    device = get_device()
    models = load_paper_models(device, include_variants=True)
    rows = []
    for name, (model, _) in models.items():
        seed_values = [
            evaluate_model(model, args.data, seed, device) for seed in EVAL_SEEDS
        ]
        row = {
            "model": name,
            "mean": float(np.mean(seed_values)),
            "std_across_seeds": float(np.std(seed_values, ddof=0)),
        }
        row.update(
            {f"seed_{seed}": value for seed, value in zip(EVAL_SEEDS, seed_values)}
        )
        rows.append(row)
        print(f"{name}: {row['mean']:.4f}%")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("mean", ascending=False).to_csv(
        args.out, index=False
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
