"""Inference-time WIPT ablations: query weighting, distance, and multi-prototype scoring."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from configs.experiment import (
    CROSS_DOMAIN_EPISODES_PER_SEED,
    DEFAULT_DATA_PATHS,
    EVAL_SEEDS,
    N_SHOT,
    N_WAY,
)
from eval.runtime import episode_loader, get_device, load_reference_wipt
from utils.metrics import accuracy, mean_ci95

DOMAINS = ("CUB", "EuroSAT", "ISIC")
TEMPERATURES = (1.0, 0.5, 0.25)


def transformed_episode(model, support_images, query_images):
    support = model.encode(support_images)
    query = model.encode(query_images)
    _, transformed_query, transformed_support, prototypes = (
        model.forward_from_embeddings(support, query, N_WAY, N_SHOT)
    )
    return transformed_query, transformed_support, prototypes


def mean_scores(query, transformed_support, metric="euclidean"):
    prototypes = transformed_support.mean(dim=2)
    if metric == "euclidean":
        return -torch.cdist(query.unsqueeze(1), prototypes).squeeze(1)
    q = F.normalize(query, dim=1).unsqueeze(1)
    p = F.normalize(prototypes, dim=2)
    return (q * p).sum(dim=2)


def multi_scores(query, transformed_support, metric="euclidean"):
    n_query = query.shape[0]
    prototypes = transformed_support.view(n_query, N_WAY * N_SHOT, -1)
    if metric == "euclidean":
        distances = torch.cdist(query.unsqueeze(1), prototypes).squeeze(1)
        return -distances.view(n_query, N_WAY, N_SHOT).min(dim=2).values
    q = F.normalize(query, dim=1).unsqueeze(1)
    p = F.normalize(prototypes, dim=2)
    similarity = (q * p).sum(dim=2).view(n_query, N_WAY, N_SHOT)
    return similarity.max(dim=2).values


def query_weighted_scores(model, support_images, query_images, temperature):
    support = model.encode(support_images)
    query = model.encode(query_images)
    n_query = query.shape[0]
    tokens = torch.cat(
        [query.unsqueeze(1), support.unsqueeze(0).expand(n_query, -1, -1)], dim=1
    )
    for layer in model.transformer_layers[:-1]:
        tokens = layer(tokens)
    last = model.transformer_layers[-1]
    residual = tokens
    normalized = last.norm1(tokens)
    attended, attention = last.attn(
        normalized,
        normalized,
        normalized,
        need_weights=True,
        average_attn_weights=True,
    )
    tokens = residual + attended
    tokens = tokens + last.mlp(last.norm2(tokens))
    tokens = model.norm(tokens)

    transformed_query = tokens[:, 0]
    transformed_support = tokens[:, 1:].view(n_query, N_WAY, N_SHOT, -1)
    query_attention = attention[:, 0, 1:].view(n_query, N_WAY, N_SHOT)
    weights = (query_attention / temperature).softmax(dim=2)
    prototypes = (transformed_support * weights.unsqueeze(-1)).sum(dim=2)
    return -torch.cdist(transformed_query.unsqueeze(1), prototypes).squeeze(1)


def evaluate_variant(model, data_path, seed, device, variant):
    values = []
    loader = episode_loader(data_path, seed, CROSS_DOMAIN_EPISODES_PER_SEED)
    with torch.no_grad():
        for support, _, query, labels in loader:
            support, query, labels = (
                support.to(device),
                query.to(device),
                labels.to(device),
            )
            if variant.startswith("query_weighted_T"):
                temperature = float(variant.split("T", 1)[1])
                scores = query_weighted_scores(model, support, query, temperature)
            else:
                transformed_query, transformed_support, _ = transformed_episode(
                    model, support, query
                )
                if variant == "mean_euclid":
                    scores = mean_scores(
                        transformed_query, transformed_support, "euclidean"
                    )
                elif variant == "mean_cosine":
                    scores = mean_scores(
                        transformed_query, transformed_support, "cosine"
                    )
                elif variant == "multi_euclid":
                    scores = multi_scores(
                        transformed_query, transformed_support, "euclidean"
                    )
                elif variant == "multi_cosine":
                    scores = multi_scores(
                        transformed_query, transformed_support, "cosine"
                    )
                else:
                    raise ValueError(variant)
            values.append(accuracy(scores, labels))
    return np.asarray(values) * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cub", default=DEFAULT_DATA_PATHS["CUB"])
    parser.add_argument("--eurosat", default=DEFAULT_DATA_PATHS["EuroSAT"])
    parser.add_argument("--isic", default=DEFAULT_DATA_PATHS["ISIC"])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/raw/ablations/inference_variants.csv"),
    )
    args = parser.parse_args()

    device = get_device()
    model, _ = load_reference_wipt(device)
    paths = {"CUB": args.cub, "EuroSAT": args.eurosat, "ISIC": args.isic}
    variants = [
        "mean_euclid",
        "mean_cosine",
        "multi_euclid",
        "multi_cosine",
        *[f"query_weighted_T{temperature}" for temperature in TEMPERATURES],
    ]

    rows = []
    for domain in DOMAINS:
        for variant in variants:
            values = np.concatenate(
                [
                    evaluate_variant(model, paths[domain], seed, device, variant)
                    for seed in EVAL_SEEDS
                ]
            )
            mean, ci = mean_ci95(values)
            rows.append(
                {
                    "dataset": domain,
                    "variant": variant,
                    "accuracy": mean,
                    "ci95": ci,
                    "n_episodes": len(values),
                }
            )
            print(f"{domain}: {variant} = {mean:.4f}%")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"saved {args.out}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
