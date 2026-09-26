"""Cross-domain 5-way 5-shot evaluation and prototype-geometry analysis."""

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
    N_SHOT,
    N_WAY,
)
from eval.runtime import episode_loader, get_device, load_paper_models
from utils.metrics import accuracy, mean_ci95

DOMAINS = ("CUB", "EuroSAT", "ISIC")
GEOMETRY_MODELS = {"ProtoNet+ViT", "SupportTransformer+ViT", "WIPT-2"}


def prototype_geometry(prototypes, query_repr, query_labels, support_repr):
    inter = []
    for first in range(N_WAY):
        for second in range(first + 1, N_WAY):
            inter.append((prototypes[first] - prototypes[second]).norm().item())
    inter_mean = float(np.mean(inter))
    support_by_class = support_repr.view(N_WAY, N_SHOT, -1)
    intra = float(
        (support_by_class - prototypes.unsqueeze(1)).norm(dim=2).mean().item()
    )
    separation = inter_mean / (intra + 1e-9)

    distances = torch.cdist(query_repr, prototypes)
    labels = query_labels.detach().cpu().numpy()
    margins = []
    for index, label in enumerate(labels):
        correct = distances[index, label].item()
        wrong = distances[index].clone()
        wrong[label] = float("inf")
        margins.append(wrong.min().item() - correct)
    margin = float(np.mean(margins))
    return separation, margin / (inter_mean + 1e-9)


def geometry_for(model, kind, support_images, query_images, query_labels):
    with torch.no_grad():
        support = model.encode(support_images)
        query = model.encode(query_images)

        if kind == "protonet":
            prototypes = support.view(N_WAY, N_SHOT, -1).mean(dim=1)
            return prototype_geometry(prototypes, query, query_labels, support)

        if kind == "support_transformer":
            transformed = support.unsqueeze(0)
            for layer in model.transformer_layers:
                transformed = layer(transformed)
            transformed = model.norm(transformed).squeeze(0)
            prototypes = transformed.view(N_WAY, N_SHOT, -1).mean(dim=1)
            return prototype_geometry(prototypes, query, query_labels, transformed)

        if kind == "wipt":
            (
                _,
                transformed_query,
                transformed_support,
                per_query_prototypes,
            ) = model.forward_from_embeddings(support, query, N_WAY, N_SHOT)
            # The paper's episode-level geometry averages the query-conditioned
            # support/prototype representations across queries before measuring
            # class separation, while retaining each transformed query.
            support_repr = transformed_support.reshape(
                transformed_support.shape[0], N_WAY * N_SHOT, -1
            ).mean(dim=0)
            prototypes = per_query_prototypes.mean(dim=0)
            return prototype_geometry(
                prototypes, transformed_query, query_labels, support_repr
            )

    return float("nan"), float("nan")


def evaluate_domain(domain, data_path, models, device):
    rows = []
    for model_name, (model, kind) in models.items():
        episode_index = 0
        for seed in EVAL_SEEDS:
            loader = episode_loader(data_path, seed, CROSS_DOMAIN_EPISODES_PER_SEED)
            with torch.no_grad():
                for support, _, query, labels in loader:
                    support, query, labels = (
                        support.to(device),
                        query.to(device),
                        labels.to(device),
                    )
                    scores = model(support, query, N_WAY, N_SHOT)
                    separation, margin_norm = (float("nan"), float("nan"))
                    if model_name in GEOMETRY_MODELS:
                        separation, margin_norm = geometry_for(
                            model, kind, support, query, labels
                        )
                    rows.append(
                        {
                            "dataset": domain,
                            "model": model_name,
                            "episode": episode_index,
                            "accuracy": accuracy(scores, labels),
                            "separation_ratio": separation,
                            "margin_norm": margin_norm,
                        }
                    )
                    episode_index += 1
        print(f"{domain}: {model_name} complete")
    return pd.DataFrame(rows)


def summarize(episodes):
    summaries = []
    for (dataset, model), group in episodes.groupby(["dataset", "model"], sort=False):
        accuracy_values = group["accuracy"].to_numpy(float)
        accuracy_mean, accuracy_ci = mean_ci95(accuracy_values)
        summaries.append(
            {
                "dataset": dataset,
                "model": model,
                "accuracy": 100 * accuracy_mean,
                "acc_ci95": 100 * accuracy_ci,
                "n_episodes": len(group),
                "separation_ratio": group["separation_ratio"].mean(skipna=True),
                "margin_norm": group["margin_norm"].mean(skipna=True),
            }
        )
    return pd.DataFrame(summaries)


def paired_tests(episodes):
    rows = []
    for dataset in DOMAINS:
        wide = episodes[episodes.dataset == dataset].pivot(
            index="episode", columns="model", values="accuracy"
        )
        for competitor in ("ProtoNet+ViT", "SupportTransformer+ViT"):
            if competitor not in wide or "WIPT-2" not in wide:
                continue
            paired = wide[["WIPT-2", competitor]].dropna()
            differences = paired["WIPT-2"] - paired[competitor]
            _, ci = mean_ci95(differences)
            t_stat, p_value = stats.ttest_rel(paired["WIPT-2"], paired[competitor])
            rows.append(
                {
                    "dataset": dataset,
                    "comparison": f"WIPT-2 - {competitor}",
                    "wipt_accuracy": 100 * paired["WIPT-2"].mean(),
                    "competitor_accuracy": 100 * paired[competitor].mean(),
                    "mean_difference_pp": 100 * differences.mean(),
                    "difference_ci95_pp": 100 * ci,
                    "paired_t": float(t_stat),
                    "paired_p": float(p_value),
                    "n_episodes": len(paired),
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cub", default=DEFAULT_DATA_PATHS["CUB"])
    parser.add_argument("--eurosat", default=DEFAULT_DATA_PATHS["EuroSAT"])
    parser.add_argument("--isic", default=DEFAULT_DATA_PATHS["ISIC"])
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/raw/cross_domain")
    )
    args = parser.parse_args()

    device = get_device()
    models = load_paper_models(device, include_variants=True)
    data_paths = {"CUB": args.cub, "EuroSAT": args.eurosat, "ISIC": args.isic}
    episodes = pd.concat(
        [
            evaluate_domain(domain, data_paths[domain], models, device)
            for domain in DOMAINS
        ],
        ignore_index=True,
    )
    summary = summarize(episodes)
    tests = paired_tests(episodes)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(args.out_dir / "episodes.csv", index=False)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    tests.to_csv(args.out_dir / "paired_tests.csv", index=False)
    print(f"saved cross-domain outputs to {args.out_dir}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
