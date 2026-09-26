"""Generate the paired EuroSAT query analysis for frozen ProtoNet and WIPT."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from configs.experiment import (
    CROSS_DOMAIN_EPISODES_PER_SEED,
    DEFAULT_DATA_PATHS,
    EVAL_SEEDS,
    N_SHOT,
    N_WAY,
)
from eval.runtime import (
    episode_loader,
    get_device,
    load_frozen_protonet,
    load_reference_wipt,
)
from utils.metrics import mean_ci95


def query_margins(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(scores.shape[0], device=scores.device)
    correct = scores[rows, labels]
    wrong = scores.clone()
    wrong[rows, labels] = float("-inf")
    return correct - wrong.max(dim=1).values


def transition_table(queries: pd.DataFrame) -> pd.DataFrame:
    table = pd.crosstab(queries.proto_correct, queries.wipt_correct)
    rows = []
    for proto in (0, 1):
        row_total = int(table.loc[proto].sum())
        for wipt in (0, 1):
            count = int(table.loc[proto, wipt])
            rows.append(
                {
                    "protonet_state": "Wrong" if proto == 0 else "Correct",
                    "wipt_state": "Wrong" if wipt == 0 else "Correct",
                    "n_queries": count,
                    "row_percentage": 100 * count / row_total,
                    "all_percentage": 100 * count / len(queries),
                }
            )
    return pd.DataFrame(rows)


def margin_bins(queries: pd.DataFrame, n_bins: int = 20) -> pd.DataFrame:
    data = queries.copy()
    data["margin_bin"] = (
        pd.qcut(data.proto_margin, n_bins, labels=False, duplicates="drop") + 1
    )
    data["paired_diff_pp"] = 100.0 * (
        data.wipt_correct.astype(float) - data.proto_correct.astype(float)
    )
    rows = []
    for bin_index, group in data.groupby("margin_bin", sort=True):
        _, ci = mean_ci95(group.paired_diff_pp)
        rows.append(
            {
                "margin_bin": int(bin_index),
                "n_queries": len(group),
                "median_protonet_margin": float(group.proto_margin.median()),
                "mean_protonet_margin": float(group.proto_margin.mean()),
                "protonet_accuracy": 100 * float(group.proto_correct.mean()),
                "wipt_accuracy": 100 * float(group.wipt_correct.mean()),
                "wipt_minus_protonet_pp": float(group.paired_diff_pp.mean()),
                "difference_ci95_pp": float(ci),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA_PATHS["EuroSAT"])
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/raw/query_analysis")
    )
    args = parser.parse_args()

    device = get_device()
    protonet = load_frozen_protonet(device)
    wipt, _ = load_reference_wipt(device)
    rows = []
    row_index = 0

    with torch.no_grad():
        for seed in EVAL_SEEDS:
            loader = episode_loader(args.data, seed, CROSS_DOMAIN_EPISODES_PER_SEED)
            for episode_index, (support, _, query, labels) in enumerate(loader):
                support = support.to(device)
                query = query.to(device)
                labels = labels.to(device)

                proto_scores = protonet(support, query, N_WAY, N_SHOT)
                wipt_scores = wipt(support, query, N_WAY, N_SHOT)
                proto_predictions = proto_scores.argmax(dim=1)
                wipt_predictions = wipt_scores.argmax(dim=1)
                proto_margin = query_margins(proto_scores, labels)
                wipt_margin = query_margins(wipt_scores, labels)

                for query_index in range(labels.shape[0]):
                    proto_correct = int(
                        proto_predictions[query_index] == labels[query_index]
                    )
                    wipt_correct = int(
                        wipt_predictions[query_index] == labels[query_index]
                    )
                    outcome = {
                        (0, 0): "both_wrong",
                        (0, 1): "wipt_rescue",
                        (1, 0): "wipt_break",
                        (1, 1): "both_correct",
                    }[(proto_correct, wipt_correct)]
                    rows.append(
                        {
                            "row_index": row_index,
                            "seed": seed,
                            "episode": episode_index,
                            "query_index": query_index,
                            "query_class": int(labels[query_index]),
                            "proto_correct": proto_correct,
                            "proto_margin": float(proto_margin[query_index]),
                            "wipt_correct": wipt_correct,
                            "wipt_margin": float(wipt_margin[query_index]),
                            "outcome": outcome,
                        }
                    )
                    row_index += 1
            print(f"EuroSAT seed {seed} complete")

    queries = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    queries.to_csv(args.out_dir / "eurosat_queries.csv", index=False)
    transition_table(queries).to_csv(
        args.out_dir / "eurosat_transitions.csv", index=False
    )
    margin_bins(queries).to_csv(args.out_dir / "eurosat_margin_bins.csv", index=False)
    print(f"saved query-analysis outputs to {args.out_dir}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
