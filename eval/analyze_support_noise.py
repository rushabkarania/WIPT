"""Analyze how Gaussian corruption of the support set affects ProtoNet and WIPT.

Query images remain clean. For WIPT, the noisy-support comparison holds the
transformed query representation from the corresponding clean-support episode
fixed while replacing only the query-conditioned prototype side. This matches
the support-pathway isolation reported in the paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from configs.experiment import (
    DEFAULT_DATA_PATHS,
    EVAL_SEEDS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    N_SHOT,
    N_WAY,
    NOISE_EPISODES_PER_SEED,
)
from eval.runtime import (
    episode_loader,
    get_device,
    load_frozen_protonet,
    load_reference_wipt,
)
from utils.metrics import accuracy

SEVERITIES = (3, 5)
GAUSSIAN_SIGMA = (0.08, 0.12, 0.18, 0.26, 0.38)
METRICS = (
    "input_shift",
    "prototype_shift",
    "noise_amplification_ratio",
    "intra_clean",
    "intra_noisy",
    "accuracy_clean",
    "accuracy_noisy",
    "margin_clean",
    "margin_noisy",
)


def add_noise(normalized_tensor: torch.Tensor, severity: int) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=normalized_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=normalized_tensor.device).view(1, 3, 1, 1)
    image = normalized_tensor * std + mean
    image = torch.clamp(
        image + torch.randn_like(image) * GAUSSIAN_SIGMA[severity - 1],
        0,
        1,
    )
    return (image - mean) / std


def mean_margin(scores: torch.Tensor, labels: torch.Tensor) -> float:
    rows = torch.arange(scores.shape[0], device=scores.device)
    correct = scores[rows, labels]
    wrong = scores.clone()
    wrong[rows, labels] = float("-inf")
    return float((correct - wrong.max(dim=1).values).mean())


def wipt_representations(model, support_images, query_images):
    support = model.encode(support_images)
    query = model.encode(query_images)
    _, transformed_query, transformed_support, prototypes = (
        model.forward_from_embeddings(
            support,
            query,
            N_WAY,
            N_SHOT,
        )
    )
    return transformed_query, transformed_support, prototypes


def empty_store():
    return {
        severity: {
            model: {metric: [] for metric in METRICS} for model in ("ProtoNet", "WIPT")
        }
        for severity in SEVERITIES
    }


def record_protonet(
    store, severity, support_clean, support_noisy, query, prototypes_clean, labels
):
    prototypes_noisy = support_noisy.view(N_WAY, N_SHOT, -1).mean(dim=1)
    input_shift = (support_noisy - support_clean).norm(dim=1).mean().item()
    prototype_shift = (prototypes_noisy - prototypes_clean).norm(dim=1).mean().item()
    clean_scores = -torch.cdist(query, prototypes_clean)
    noisy_scores = -torch.cdist(query, prototypes_noisy)

    target = store[severity]["ProtoNet"]
    target["input_shift"].append(input_shift)
    target["prototype_shift"].append(prototype_shift)
    target["noise_amplification_ratio"].append(prototype_shift / (input_shift + 1e-8))
    target["intra_clean"].append(
        (support_clean.view(N_WAY, N_SHOT, -1) - prototypes_clean.unsqueeze(1))
        .norm(dim=2)
        .mean()
        .item()
    )
    target["intra_noisy"].append(
        (support_noisy.view(N_WAY, N_SHOT, -1) - prototypes_noisy.unsqueeze(1))
        .norm(dim=2)
        .mean()
        .item()
    )
    target["accuracy_clean"].append(accuracy(clean_scores, labels))
    target["accuracy_noisy"].append(accuracy(noisy_scores, labels))
    target["margin_clean"].append(mean_margin(clean_scores, labels))
    target["margin_noisy"].append(mean_margin(noisy_scores, labels))


def record_wipt(
    store,
    severity,
    query_clean,
    support_clean,
    support_noisy,
    prototypes_clean,
    prototypes_noisy,
    labels,
):
    input_shift = (support_noisy - support_clean).norm(dim=3).mean().item()
    prototype_shift = (prototypes_noisy - prototypes_clean).norm(dim=2).mean().item()
    clean_scores = -torch.cdist(query_clean.unsqueeze(1), prototypes_clean).squeeze(1)
    noisy_scores = -torch.cdist(query_clean.unsqueeze(1), prototypes_noisy).squeeze(1)

    target = store[severity]["WIPT"]
    target["input_shift"].append(input_shift)
    target["prototype_shift"].append(prototype_shift)
    target["noise_amplification_ratio"].append(prototype_shift / (input_shift + 1e-8))
    target["intra_clean"].append(
        (support_clean - prototypes_clean.unsqueeze(2)).norm(dim=3).mean().item()
    )
    target["intra_noisy"].append(
        (support_noisy - prototypes_noisy.unsqueeze(2)).norm(dim=3).mean().item()
    )
    target["accuracy_clean"].append(accuracy(clean_scores, labels))
    target["accuracy_noisy"].append(accuracy(noisy_scores, labels))
    target["margin_clean"].append(mean_margin(clean_scores, labels))
    target["margin_noisy"].append(mean_margin(noisy_scores, labels))


def summarize_store(store) -> pd.DataFrame:
    rows = []
    for severity in SEVERITIES:
        for model in ("ProtoNet", "WIPT"):
            rows.append(
                {
                    "severity": severity,
                    "model": model,
                    **{
                        metric: float(np.mean(store[severity][model][metric]))
                        for metric in METRICS
                    },
                }
            )
    return pd.DataFrame(rows)


def figure_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in ("ProtoNet", "WIPT"):
        clean = metrics[metrics.model == model].iloc[0]
        rows.append(
            {
                "severity": 0,
                "condition": "Clean",
                "model": model,
                "noise_amplification_ratio": "not_applicable",
                "absolute_margin": clean.margin_clean,
                "accuracy": 100 * clean.accuracy_clean,
            }
        )
        for severity in SEVERITIES:
            row = metrics[
                (metrics.model == model) & (metrics.severity == severity)
            ].iloc[0]
            rows.append(
                {
                    "severity": severity,
                    "condition": f"Severity {severity}",
                    "model": model,
                    "noise_amplification_ratio": row.noise_amplification_ratio,
                    "absolute_margin": row.margin_noisy,
                    "accuracy": 100 * row.accuracy_noisy,
                }
            )
    return pd.DataFrame(rows).sort_values(["severity", "model"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA_PATHS["miniImageNet_test"])
    parser.add_argument("--out-dir", type=Path, default=Path("results/raw/noise"))
    args = parser.parse_args()

    device = get_device()
    protonet = load_frozen_protonet(device)
    wipt, _ = load_reference_wipt(device)
    store = empty_store()

    for seed in EVAL_SEEDS:
        loader = episode_loader(args.data, seed, NOISE_EPISODES_PER_SEED)
        for support_images, _, query_images, labels in loader:
            support_images = support_images.to(device)
            query_images = query_images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                proto_support_clean = protonet.encode(support_images)
                proto_query = protonet.encode(query_images)
                proto_clean = proto_support_clean.view(N_WAY, N_SHOT, -1).mean(dim=1)

                wipt_query_clean, wipt_support_clean, wipt_proto_clean = (
                    wipt_representations(
                        wipt,
                        support_images,
                        query_images,
                    )
                )

                for severity in SEVERITIES:
                    noisy_support_images = add_noise(support_images, severity)

                    proto_support_noisy = protonet.encode(noisy_support_images)
                    record_protonet(
                        store,
                        severity,
                        proto_support_clean,
                        proto_support_noisy,
                        proto_query,
                        proto_clean,
                        labels,
                    )

                    _, wipt_support_noisy, wipt_proto_noisy = wipt_representations(
                        wipt,
                        noisy_support_images,
                        query_images,
                    )
                    record_wipt(
                        store,
                        severity,
                        wipt_query_clean,
                        wipt_support_clean,
                        wipt_support_noisy,
                        wipt_proto_clean,
                        wipt_proto_noisy,
                        labels,
                    )
        print(f"miniImageNet seed {seed} complete")

    metrics = summarize_store(store)
    summary = figure_summary(metrics)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.out_dir / "support_noise_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "support_noise_summary.csv", index=False)
    print(f"saved support-noise outputs to {args.out_dir}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
