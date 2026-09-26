"""Mechanistic diagnostics for why WIPT gains depend on shot count and target domain.

The analysis uses query labels only after prediction, as an analysis-only oracle.  In
particular, the labeled query centroid measures how noisy the support-derived class
representative is in the frozen embedding space.  It is never used by a classifier.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
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
    shot_tag,
    t_ci95,
)


def interclass_mean(points: torch.Tensor) -> float:
    values = []
    for i in range(points.shape[0]):
        for j in range(i + 1, points.shape[0]):
            values.append(float((points[i] - points[j]).norm()))
    return float(np.mean(values)) if values else float("nan")


def episode_oracle_metrics(
    support: torch.Tensor, query: torch.Tensor, labels: torch.Tensor, n_shot: int
) -> dict[str, float]:
    support_proto = support.view(N_WAY, n_shot, -1).mean(1)
    query_centroids = torch.stack([query[labels == c].mean(0) for c in range(N_WAY)])
    scale = interclass_mean(query_centroids) + 1e-12
    centroid_error = float((support_proto - query_centroids).norm(dim=1).mean()) / scale
    query_scatter = (
        float(
            torch.stack(
                [
                    (query[labels == c] - query_centroids[c]).norm(dim=1).mean()
                    for c in range(N_WAY)
                ]
            ).mean()
        )
        / scale
    )
    support_separation = interclass_mean(support_proto) / scale
    return {
        "support_to_query_centroid_error_norm": centroid_error,
        "query_within_class_scatter_norm": query_scatter,
        "support_interclass_separation_norm": support_separation,
        "query_centroid_interclass_distance": scale,
    }


def safe_corr(x, y, method: str) -> tuple[float, float]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return float("nan"), float("nan")
    if method == "pearson":
        r, p = stats.pearsonr(x[mask], y[mask])
    else:
        r, p = stats.spearmanr(x[mask], y[mask])
    return float(r), float(p)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot", type=int, required=True, choices=[1, 5])
    parser.add_argument("--seeds", type=int, nargs="*", default=list(TRAIN_SEEDS))
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("checkpoints/seed_study")
    )
    parser.add_argument(
        "--out-root", type=Path, default=Path("results/raw/shot_domain_mechanism")
    )
    parser.add_argument(
        "--episodes-per-seed", type=int, default=CROSS_DOMAIN_EPISODES_PER_SEED
    )
    parser.add_argument("--encoder-batch-size", type=int, default=64)
    parser.add_argument("--cub", default=DEFAULT_DATA_PATHS["CUB"])
    parser.add_argument("--eurosat", default=DEFAULT_DATA_PATHS["EuroSAT"])
    parser.add_argument("--isic", default=DEFAULT_DATA_PATHS["ISIC"])
    parser.add_argument(
        "--extra-dataset", action="append", default=[], help="NAME=PATH"
    )
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    device = get_device()
    encoder, wipt_heads, _ = load_seed_study_heads(
        args.checkpoint_root, n_shot=args.shot, training_seeds=seeds, device=device
    )
    datasets = {"CUB": args.cub, "EuroSAT": args.eurosat, "ISIC": args.isic}
    for item in args.extra_dataset:
        if "=" not in item:
            raise ValueError("--extra-dataset must be NAME=PATH")
        name, path = item.split("=", 1)
        datasets[name] = path

    rows = []
    with torch.no_grad():
        for dataset, path in datasets.items():
            global_episode = 0
            for eval_seed in EVAL_SEEDS:
                loader = episode_loader(
                    path,
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
                    s = encode_in_chunks(
                        encoder, support, chunk_size=args.encoder_batch_size
                    )
                    q = encode_in_chunks(
                        encoder, query, chunk_size=args.encoder_batch_size
                    )
                    oracle = episode_oracle_metrics(s, q, labels, args.shot)
                    pn = protonet_scores(s, q, N_WAY, args.shot)
                    pn_pred = pn.argmax(1)
                    pn_correct = pn_pred.eq(labels)
                    pn_accuracy = 100.0 * float(pn_correct.float().mean())
                    pn_margin = query_margin(pn, labels)
                    support_proto = s.view(N_WAY, args.shot, -1).mean(1)
                    scale = oracle["query_centroid_interclass_distance"]

                    for seed in seeds:
                        scores, _, transformed_support, prototypes = wipt_heads[
                            seed
                        ].forward_from_embeddings(s, q, N_WAY, args.shot)
                        pred = scores.argmax(1)
                        correct = pred.eq(labels)
                        wipt_accuracy = 100.0 * float(correct.float().mean())
                        rescued = (~pn_correct) & correct
                        broken = pn_correct & (~correct)
                        flip = pred.ne(pn_pred)

                        # Query-conditioned prototype variability: how much the same class
                        # representative changes as the query changes.
                        proto_center = prototypes.mean(dim=0, keepdim=True)
                        conditioning_variability = float(
                            (prototypes - proto_center).norm(dim=2).mean()
                        ) / (scale + 1e-12)
                        # Total displacement away from the raw frozen class means.  Spaces differ,
                        # so this is descriptive rather than a direct "toward the query" metric.
                        raw = support_proto.unsqueeze(0).expand_as(prototypes)
                        prototype_displacement = float(
                            (prototypes - raw).norm(dim=2).mean()
                        ) / (scale + 1e-12)

                        row = {
                            "dataset": dataset,
                            "n_shot": args.shot,
                            "training_seed": seed,
                            "eval_seed": eval_seed,
                            "episode_in_seed": episode_in_seed,
                            "episode": global_episode,
                            "protonet_accuracy": pn_accuracy,
                            "wipt_accuracy": wipt_accuracy,
                            "wipt_minus_protonet_pp": wipt_accuracy - pn_accuracy,
                            "protonet_mean_true_margin": float(pn_margin.mean()),
                            "protonet_margin_sd": float(pn_margin.std(unbiased=True)),
                            "prediction_flip_rate": 100.0 * float(flip.float().mean()),
                            "rescue_rate_among_pn_errors": 100.0
                            * float(rescued.sum())
                            / max(int((~pn_correct).sum()), 1),
                            "break_rate_among_pn_successes": 100.0
                            * float(broken.sum())
                            / max(int(pn_correct.sum()), 1),
                            "conditioning_variability_norm": conditioning_variability,
                            "prototype_displacement_norm": prototype_displacement,
                            **oracle,
                        }
                        rows.append(row)
                    global_episode += 1
            print(f"{dataset}: mechanism analysis complete")

    frame = pd.DataFrame(rows)
    summary_rows = []
    metrics = [
        "wipt_minus_protonet_pp",
        "support_to_query_centroid_error_norm",
        "query_within_class_scatter_norm",
        "support_interclass_separation_norm",
        "protonet_mean_true_margin",
        "prediction_flip_rate",
        "rescue_rate_among_pn_errors",
        "break_rate_among_pn_successes",
        "conditioning_variability_norm",
        "prototype_displacement_norm",
    ]
    # First average matched episodes within each independently trained WIPT head,
    # then report variation across heads.
    run_means = frame.groupby(["dataset", "training_seed"], as_index=False)[
        metrics
    ].mean()
    for dataset, group in run_means.groupby("dataset", sort=False):
        for metric in metrics:
            mean, sd, ci = t_ci95(group[metric])
            summary_rows.append(
                {
                    "dataset": dataset,
                    "n_shot": args.shot,
                    "metric": metric,
                    "mean": mean,
                    "training_seed_sd": sd,
                    "training_seed_ci95": ci,
                }
            )

    corr_rows = []
    predictors = [
        "support_to_query_centroid_error_norm",
        "query_within_class_scatter_norm",
        "support_interclass_separation_norm",
        "protonet_mean_true_margin",
        "conditioning_variability_norm",
        "prototype_displacement_norm",
    ]
    for (dataset, seed), group in frame.groupby(
        ["dataset", "training_seed"], sort=False
    ):
        for predictor in predictors:
            for method in ["pearson", "spearman"]:
                r, p = safe_corr(
                    group[predictor], group["wipt_minus_protonet_pp"], method
                )
                corr_rows.append(
                    {
                        "dataset": dataset,
                        "n_shot": args.shot,
                        "training_seed": seed,
                        "predictor": predictor,
                        "method": method,
                        "correlation": r,
                        "p_value": p,
                        "n_episodes": len(group),
                    }
                )

    out = args.out_root / shot_tag(args.shot)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "episode_metrics.csv", index=False)
    run_means.to_csv(out / "run_means.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out / "domain_summary.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(out / "correlations.csv", index=False)
    print(f"saved mechanism analysis to {out}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
