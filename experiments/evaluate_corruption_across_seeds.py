"""Evaluate EuroSAT A/B/C corruption behavior across independent training seeds."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from configs.experiment import (
    NUM_WORKERS,
    CORRUPTION_EPISODES_PER_SEED,
    DEFAULT_DATA_PATHS,
    EVAL_SEEDS,
    N_QUERY,
    N_WAY,
    TRAIN_SEEDS,
)
from eval.evaluate_corruptions import CATEGORIES, SEVERITIES, CorruptedDataset
from eval.runtime import get_device
from experiments.seed_study_common import (
    accuracy_percent,
    encode_in_chunks,
    load_seed_study_heads,
    parse_seeds,
    protonet_scores,
    seed_result_dir,
    t_ci95,
)
from utils.data import EpisodicCollate, EpisodicSampler
from utils.metrics import mean_ci95, set_seed


def make_corruption_loader(
    data_path,
    corruption_names,
    severity,
    eval_seed,
    *,
    n_shot,
    n_episodes,
):
    set_seed(eval_seed)
    dataset = CorruptedDataset(data_path, corruption_names, max(severity, 1))
    sampler = EpisodicSampler(
        dataset.labels,
        N_WAY,
        n_shot,
        N_QUERY,
        n_episodes,
        seed=eval_seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=EpisodicCollate(N_WAY, n_shot, N_QUERY),
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


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
    parser.add_argument("--data", default=DEFAULT_DATA_PATHS["EuroSAT"])
    parser.add_argument(
        "--episodes-per-seed",
        type=int,
        default=CORRUPTION_EPISODES_PER_SEED,
    )
    parser.add_argument("--encoder-batch-size", type=int, default=64)
    args = parser.parse_args()

    training_seeds = parse_seeds(args.seeds)
    device = get_device()
    encoder, wipt_heads, support_heads = load_seed_study_heads(
        args.checkpoint_root,
        n_shot=args.shot,
        training_seeds=training_seeds,
        device=device,
    )

    conditions = [("clean", 0, ())]
    conditions.extend(
        (category, severity, corruption_names)
        for category, corruption_names in CATEGORIES.items()
        for severity in SEVERITIES
    )

    run_rows = []
    gap_rows = []
    with torch.no_grad():
        for category, severity, corruption_names in conditions:
            values = defaultdict(list)
            for eval_seed in EVAL_SEEDS:
                loader = make_corruption_loader(
                    args.data,
                    corruption_names,
                    severity,
                    eval_seed,
                    n_shot=args.shot,
                    n_episodes=args.episodes_per_seed,
                )
                for support, _, query, labels in loader:
                    support = support.to(device)
                    query = query.to(device)
                    labels = labels.to(device)
                    support_embeddings = encode_in_chunks(
                        encoder, support, chunk_size=args.encoder_batch_size
                    )
                    query_embeddings = encode_in_chunks(
                        encoder, query, chunk_size=args.encoder_batch_size
                    )
                    proto_scores = protonet_scores(
                        support_embeddings,
                        query_embeddings,
                        N_WAY,
                        args.shot,
                    )
                    values[("ProtoNet", "fixed")].append(
                        accuracy_percent(proto_scores, labels)
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
                        values[("WIPT", training_seed)].append(
                            accuracy_percent(wipt_scores, labels)
                        )
                        values[("Support-only Transformer", training_seed)].append(
                            accuracy_percent(support_scores, labels)
                        )

            proto_mean = float(pd.Series(values[("ProtoNet", "fixed")]).mean())
            support_means = {}
            wipt_means = {}
            for (model, training_seed), episode_values in values.items():
                mean, ci = mean_ci95(episode_values)
                run_rows.append(
                    {
                        "dataset": "EuroSAT",
                        "category": category,
                        "severity": severity,
                        "model": model,
                        "training_seed": training_seed,
                        "accuracy": mean,
                        "episode_ci95": ci,
                        "n_episodes": len(episode_values),
                    }
                )
                if model == "WIPT":
                    wipt_means[int(training_seed)] = mean
                elif model == "Support-only Transformer":
                    support_means[int(training_seed)] = mean

            for training_seed in training_seeds:
                gap_rows.extend(
                    [
                        {
                            "dataset": "EuroSAT",
                            "category": category,
                            "severity": severity,
                            "training_seed": training_seed,
                            "comparison": "WIPT - ProtoNet",
                            "difference_pp": wipt_means[training_seed] - proto_mean,
                        },
                        {
                            "dataset": "EuroSAT",
                            "category": category,
                            "severity": severity,
                            "training_seed": training_seed,
                            "comparison": "WIPT - Support-only Transformer",
                            "difference_pp": (
                                wipt_means[training_seed] - support_means[training_seed]
                            ),
                        },
                    ]
                )
            print(f"EuroSAT: {category} severity {severity} complete")

    runs = pd.DataFrame(run_rows)
    gaps = pd.DataFrame(gap_rows)
    summary_rows = []
    for (category, severity, comparison), group in gaps.groupby(
        ["category", "severity", "comparison"], sort=False
    ):
        mean, sd, ci = t_ci95(group.difference_pp)
        summary_rows.append(
            {
                "dataset": "EuroSAT",
                "category": category,
                "severity": severity,
                "comparison": comparison,
                "mean_difference_pp": mean,
                "training_seed_sd_pp": sd,
                "training_seed_ci95_pp": ci,
                "n_training_seeds": len(group),
                "positive_runs": int((group.difference_pp > 0).sum()),
                "negative_runs": int((group.difference_pp < 0).sum()),
            }
        )

    output = seed_result_dir(args.out_root, args.shot) / "corruption"
    output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output / "corruption_runs.csv", index=False)
    gaps.to_csv(output / "corruption_gap_by_training_seed.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(
        output / "corruption_gap_summary.csv", index=False
    )
    print(f"saved corruption-stability outputs to {output}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
