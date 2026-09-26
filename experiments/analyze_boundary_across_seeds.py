"""Test whether EuroSAT boundary-local behavior persists across WIPT training seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

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
    encode_in_chunks,
    load_seed_study_heads,
    parse_seeds,
    protonet_scores,
    query_margin,
    seed_result_dir,
    t_ci95,
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
        default=CROSS_DOMAIN_EPISODES_PER_SEED,
    )
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--encoder-batch-size", type=int, default=64)
    args = parser.parse_args()

    training_seeds = parse_seeds(args.seeds)
    device = get_device()
    encoder, wipt_heads, _ = load_seed_study_heads(
        args.checkpoint_root,
        n_shot=args.shot,
        training_seeds=training_seeds,
        device=device,
    )

    rows = []
    query_id = 0
    with torch.no_grad():
        for eval_seed in EVAL_SEEDS:
            loader = episode_loader(
                args.data,
                eval_seed,
                args.episodes_per_seed,
                n_way=N_WAY,
                n_shot=args.shot,
                n_query=N_QUERY,
            )
            for episode, (support, _, query, labels) in enumerate(loader):
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
                proto_pred = proto_scores.argmax(dim=1)
                proto_margin = query_margin(proto_scores, labels)

                wipt_correct = {}
                for training_seed in training_seeds:
                    scores, _, _, _ = wipt_heads[training_seed].forward_from_embeddings(
                        support_embeddings,
                        query_embeddings,
                        N_WAY,
                        args.shot,
                    )
                    wipt_correct[training_seed] = (
                        (scores.argmax(dim=1) == labels).int().cpu()
                    )

                proto_correct = (proto_pred == labels).int().cpu()
                margins = proto_margin.cpu()
                labels_cpu = labels.cpu()
                for index in range(labels.shape[0]):
                    row = {
                        "query_id": query_id,
                        "eval_seed": eval_seed,
                        "episode": episode,
                        "query_index": index,
                        "query_class": int(labels_cpu[index]),
                        "protonet_correct": int(proto_correct[index]),
                        "protonet_margin": float(margins[index]),
                    }
                    for training_seed in training_seeds:
                        row[f"wipt_correct_seed_{training_seed}"] = int(
                            wipt_correct[training_seed][index]
                        )
                    rows.append(row)
                    query_id += 1
            print(f"EuroSAT eval seed {eval_seed}: boundary analysis complete")

    queries = pd.DataFrame(rows)
    queries["margin_bin"] = (
        pd.qcut(queries.protonet_margin, args.bins, labels=False, duplicates="drop") + 1
    )

    transition_rows = []
    bin_rows = []
    for training_seed in training_seeds:
        column = f"wipt_correct_seed_{training_seed}"
        proto_wrong = queries.protonet_correct == 0
        proto_right = ~proto_wrong
        rescued = int((proto_wrong & (queries[column] == 1)).sum())
        broken = int((proto_right & (queries[column] == 0)).sum())
        transition_rows.append(
            {
                "training_seed": training_seed,
                "n_queries": len(queries),
                "protonet_errors": int(proto_wrong.sum()),
                "protonet_successes": int(proto_right.sum()),
                "rescued": rescued,
                "broken": broken,
                "rescue_rate_percent": 100.0 * rescued / max(int(proto_wrong.sum()), 1),
                "break_rate_percent": 100.0 * broken / max(int(proto_right.sum()), 1),
                "net_correct_change": rescued - broken,
            }
        )

        for margin_bin, group in queries.groupby("margin_bin", sort=True):
            diff = 100.0 * (
                group[column].astype(float) - group.protonet_correct.astype(float)
            )
            bin_rows.append(
                {
                    "training_seed": training_seed,
                    "margin_bin": int(margin_bin),
                    "n_queries": len(group),
                    "median_protonet_margin": float(group.protonet_margin.median()),
                    "protonet_accuracy": 100.0 * float(group.protonet_correct.mean()),
                    "wipt_accuracy": 100.0 * float(group[column].mean()),
                    "wipt_minus_protonet_pp": float(diff.mean()),
                }
            )

    transitions = pd.DataFrame(transition_rows)
    bins = pd.DataFrame(bin_rows)
    summary_rows = []
    for margin_bin, group in bins.groupby("margin_bin", sort=True):
        mean, sd, ci = t_ci95(group.wipt_minus_protonet_pp)
        summary_rows.append(
            {
                "margin_bin": int(margin_bin),
                "n_queries_per_training_seed": int(group.n_queries.iloc[0]),
                "median_protonet_margin": float(group.median_protonet_margin.iloc[0]),
                "mean_wipt_minus_protonet_pp": mean,
                "training_seed_sd_pp": sd,
                "training_seed_ci95_pp": ci,
                "positive_runs": int((group.wipt_minus_protonet_pp > 0).sum()),
                "negative_runs": int((group.wipt_minus_protonet_pp < 0).sum()),
            }
        )

    output = seed_result_dir(args.out_root, args.shot) / "boundary"
    output.mkdir(parents=True, exist_ok=True)
    queries.to_csv(output / "eurosat_queries_across_training_seeds.csv", index=False)
    transitions.to_csv(output / "transitions_by_training_seed.csv", index=False)
    bins.to_csv(output / "margin_bins_by_training_seed.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output / "margin_bin_summary.csv", index=False)
    print(f"saved boundary-stability outputs to {output}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
