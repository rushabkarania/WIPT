"""Compare the margin-trained WIPT checkpoint with standard WIPT."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import torch

from configs.experiment import (
    CHECKPOINTS,
    CROSS_DOMAIN_EPISODES_PER_SEED,
    DEFAULT_DATA_PATHS,
    EVAL_SEEDS,
    N_SHOT,
    N_WAY,
)
from eval.runtime import (
    episode_loader,
    get_device,
    load_reference_wipt,
    reference_encoder_state,
)
from models.wipt import WIPT
from utils.checkpoints import load_checkpoint, load_wipt_state_compat, verify_encoder
from utils.metrics import accuracy

DOMAINS = ("CUB", "EuroSAT", "ISIC")


def load_margin_model(device):
    checkpoint = load_checkpoint(CHECKPOINTS["WIPT-Margin"], device="cpu")
    model = WIPT(pretrained=False, num_layers=int(checkpoint.get("num_layers", 2)))
    load_wipt_state_compat(model, checkpoint)
    verify_encoder(model, reference_encoder_state())
    return model.to(device).eval()


def evaluate(model, data_path, seed, device):
    values = []
    with torch.no_grad():
        for support, _, query, labels in episode_loader(
            data_path, seed, CROSS_DOMAIN_EPISODES_PER_SEED
        ):
            support, query, labels = (
                support.to(device),
                query.to(device),
                labels.to(device),
            )
            values.append(accuracy(model(support, query, N_WAY, N_SHOT), labels))
    return np.asarray(values) * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cub", default=DEFAULT_DATA_PATHS["CUB"])
    parser.add_argument("--eurosat", default=DEFAULT_DATA_PATHS["EuroSAT"])
    parser.add_argument("--isic", default=DEFAULT_DATA_PATHS["ISIC"])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/raw/ablations/margin_variant.csv"),
    )
    args = parser.parse_args()

    device = get_device()
    standard, _ = load_reference_wipt(device)
    margin_model = load_margin_model(device)
    paths = {"CUB": args.cub, "EuroSAT": args.eurosat, "ISIC": args.isic}
    rows = []
    for domain in DOMAINS:
        standard_values = np.concatenate(
            [evaluate(standard, paths[domain], seed, device) for seed in EVAL_SEEDS]
        )
        margin_values = np.concatenate(
            [evaluate(margin_model, paths[domain], seed, device) for seed in EVAL_SEEDS]
        )
        difference = margin_values - standard_values
        _, p_value = stats.ttest_rel(margin_values, standard_values)
        rows.append(
            {
                "dataset": domain,
                "standard_accuracy": standard_values.mean(),
                "margin_accuracy": margin_values.mean(),
                "difference_pp": difference.mean(),
                "paired_p": p_value,
                "n_episodes": len(difference),
            }
        )
        print(f"{domain}: margin variant gap {difference.mean():+.4f} pp")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"saved {args.out}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
