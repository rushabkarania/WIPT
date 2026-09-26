"""Evaluate independently trained heads on fixed clean episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from configs.experiment import (
    CROSS_DOMAIN_EPISODES_PER_SEED,
    DEFAULT_DATA_PATHS,
    EVAL_SEEDS,
    N_QUERY,
    N_WAY,
    TRAIN_SEEDS,
)
from eval.runtime import episode_loader, get_device
from experiments.seed_study_common import (
    accuracy_percent,
    encode_in_chunks,
    load_seed_study_heads,
    parse_seeds,
    protonet_scores,
    seed_result_dir,
    t_ci95,
)
from utils.metrics import mean_ci95


def run_accuracy_summary(episode_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, model, seed), group in episode_frame.groupby(
        ["dataset", "model", "training_seed"],
        sort=False,
        dropna=False,
    ):
        mean, ci = mean_ci95(group.accuracy)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "training_seed": seed,
                "accuracy": mean,
                "episode_ci95": ci,
                "n_episodes": len(group),
            }
        )
    return pd.DataFrame(rows)


def training_seed_summary(run_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, model), group in run_frame.groupby(["dataset", "model"], sort=False):
        if model == "ProtoNet":
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "mean_accuracy": float(group.accuracy.iloc[0]),
                    "training_seed_sd": np.nan,
                    "training_seed_ci95": np.nan,
                    "n_training_seeds": 0,
                    "min_accuracy": float(group.accuracy.iloc[0]),
                    "max_accuracy": float(group.accuracy.iloc[0]),
                }
            )
            continue
        mean, sd, ci = t_ci95(group.accuracy)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "mean_accuracy": mean,
                "training_seed_sd": sd,
                "training_seed_ci95": ci,
                "n_training_seeds": len(group),
                "min_accuracy": float(group.accuracy.min()),
                "max_accuracy": float(group.accuracy.max()),
            }
        )
    return pd.DataFrame(rows)


def paired_seed_differences(run_frame: pd.DataFrame, training_seeds) -> pd.DataFrame:
    rows = []
    for dataset in run_frame.dataset.unique():
        data = run_frame[run_frame.dataset == dataset]
        proto = float(data[data.model == "ProtoNet"].accuracy.iloc[0])
        for seed in training_seeds:
            wipt = float(
                data[
                    (data.model == "WIPT") & (data.training_seed == seed)
                ].accuracy.iloc[0]
            )
            support = float(
                data[
                    (data.model == "Support-only Transformer")
                    & (data.training_seed == seed)
                ].accuracy.iloc[0]
            )
            rows.extend(
                [
                    {
                        "dataset": dataset,
                        "training_seed": seed,
                        "comparison": "WIPT - Support-only Transformer",
                        "difference_pp": wipt - support,
                    },
                    {
                        "dataset": dataset,
                        "training_seed": seed,
                        "comparison": "WIPT - ProtoNet",
                        "difference_pp": wipt - proto,
                    },
                ]
            )
    return pd.DataFrame(rows)


def difference_summary(differences: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, comparison), group in differences.groupby(
        ["dataset", "comparison"], sort=False
    ):
        mean, sd, ci = t_ci95(group.difference_pp)
        rows.append(
            {
                "dataset": dataset,
                "comparison": comparison,
                "mean_difference_pp": mean,
                "training_seed_sd_pp": sd,
                "training_seed_ci95_pp": ci,
                "n_training_seeds": len(group),
                "positive_runs": int((group.difference_pp > 0).sum()),
                "negative_runs": int((group.difference_pp < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot", type=int, default=5, choices=[1, 5])
    parser.add_argument("--seeds", type=int, nargs="*", default=list(TRAIN_SEEDS))
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
    parser.add_argument(
        "--episodes-per-seed",
        type=int,
        default=CROSS_DOMAIN_EPISODES_PER_SEED,
    )
    parser.add_argument("--encoder-batch-size", type=int, default=64)
    parser.add_argument("--include-in-domain", action="store_true")
    parser.add_argument("--cub", default=DEFAULT_DATA_PATHS["CUB"])
    parser.add_argument("--eurosat", default=DEFAULT_DATA_PATHS["EuroSAT"])
    parser.add_argument("--isic", default=DEFAULT_DATA_PATHS["ISIC"])
    parser.add_argument(
        "--miniimagenet", default=DEFAULT_DATA_PATHS["miniImageNet_test"]
    )
    parser.add_argument(
        "--dataset", action="append", default=[], help="Additional NAME=PATH target"
    )
    parser.add_argument("--skip-default-targets", action="store_true")
    args = parser.parse_args()

    training_seeds = parse_seeds(args.seeds)
    device = get_device()
    encoder, wipt_heads, support_heads = load_seed_study_heads(
        args.checkpoint_root,
        n_shot=args.shot,
        training_seeds=training_seeds,
        device=device,
    )

    datasets = {
        "CUB": args.cub,
        "EuroSAT": args.eurosat,
        "ISIC": args.isic,
    }
    if args.skip_default_targets:
        datasets = {}
    for item in args.dataset:
        name, separator, path = item.partition("=")
        if not separator or not name or not path:
            parser.error("--dataset requires NAME=PATH")
        datasets[name] = path
    if args.include_in_domain:
        datasets = {"miniImageNet": args.miniimagenet, **datasets}

    if not datasets:
        parser.error("At least one target dataset is required")
    rows = []
    with torch.no_grad():
        for dataset, data_path in datasets.items():
            global_episode = 0
            for eval_seed in EVAL_SEEDS:
                loader = episode_loader(
                    data_path,
                    eval_seed,
                    args.episodes_per_seed,
                    n_way=N_WAY,
                    n_shot=args.shot,
                    n_query=N_QUERY,
                )
                for episode_in_seed, (support, _, query, labels) in enumerate(loader):
                    support = support.to(device)
                    query = query.to(device)
                    labels = labels.to(device)
                    support_embeddings = encode_in_chunks(
                        encoder, support, chunk_size=args.encoder_batch_size
                    )
                    query_embeddings = encode_in_chunks(
                        encoder, query, chunk_size=args.encoder_batch_size
                    )

                    proto = protonet_scores(
                        support_embeddings,
                        query_embeddings,
                        N_WAY,
                        args.shot,
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "model": "ProtoNet",
                            "training_seed": "fixed",
                            "eval_seed": eval_seed,
                            "episode_in_seed": episode_in_seed,
                            "episode": global_episode,
                            "accuracy": accuracy_percent(proto, labels),
                        }
                    )

                    for training_seed in training_seeds:
                        wipt_scores, _, _, _ = wipt_heads[
                            training_seed
                        ].forward_from_embeddings(
                            support_embeddings,
                            query_embeddings,
                            N_WAY,
                            args.shot,
                        )
                        support_scores = support_heads[
                            training_seed
                        ].forward_from_embeddings(
                            support_embeddings,
                            query_embeddings,
                            N_WAY,
                            args.shot,
                        )
                        rows.extend(
                            [
                                {
                                    "dataset": dataset,
                                    "model": "WIPT",
                                    "training_seed": training_seed,
                                    "eval_seed": eval_seed,
                                    "episode_in_seed": episode_in_seed,
                                    "episode": global_episode,
                                    "accuracy": accuracy_percent(wipt_scores, labels),
                                },
                                {
                                    "dataset": dataset,
                                    "model": "Support-only Transformer",
                                    "training_seed": training_seed,
                                    "eval_seed": eval_seed,
                                    "episode_in_seed": episode_in_seed,
                                    "episode": global_episode,
                                    "accuracy": accuracy_percent(
                                        support_scores, labels
                                    ),
                                },
                            ]
                        )
                    global_episode += 1
            print(f"{dataset}: clean seed-study evaluation complete")

    episode_frame = pd.DataFrame(rows)
    numeric_seed = pd.to_numeric(episode_frame.training_seed, errors="coerce")
    episode_frame["training_seed"] = numeric_seed.where(numeric_seed.notna(), "fixed")
    run_frame = run_accuracy_summary(episode_frame)
    differences = paired_seed_differences(run_frame, training_seeds)

    output = seed_result_dir(args.out_root, args.shot) / "clean"
    output.mkdir(parents=True, exist_ok=True)
    episode_frame.to_csv(output / "episode_accuracy.csv", index=False)
    run_frame.to_csv(output / "run_accuracy.csv", index=False)
    training_seed_summary(run_frame).to_csv(
        output / "training_seed_summary.csv", index=False
    )
    differences.to_csv(output / "paired_training_seed_differences.csv", index=False)
    difference_summary(differences).to_csv(
        output / "difference_summary.csv", index=False
    )
    print(f"saved clean seed-study outputs to {output}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
