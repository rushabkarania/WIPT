"""Evaluate WIPT q=1..5 on matched cross-domain episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from configs.experiment import (
    CROSS_DOMAIN_EPISODES_PER_SEED,
    DEFAULT_DATA_PATHS,
    EVAL_SEEDS,
    N_QUERY,
    N_WAY,
    TRAIN_SEEDS,
    BACKBONE,
    WIPT_DROPOUT,
    WIPT_HEADS,
)
from eval.runtime import episode_loader, get_device
from experiments.seed_study_common import (
    encode_in_chunks,
    parse_seeds,
    seed_checkpoint_dir,
    shot_tag,
    t_ci95,
)
from models.backbone import build_frozen_encoder
from models.wipt import WIPT
from utils.checkpoints import (
    encoder_state_from_checkpoint,
    load_checkpoint,
    load_wipt_state_compat,
    verify_encoder,
)
from utils.metrics import mean_ci95


def checkpoint_for(
    q: int, seed: int, shot: int, q1_root: Path, multi_root: Path
) -> Path:
    if q == 1:
        return seed_checkpoint_dir(q1_root, shot, seed) / "wipt_l2.pth"
    return multi_root / shot_tag(shot) / f"q{q}" / f"seed_{seed}" / "wipt_l2.pth"


def load_heads(q_values, seeds, shot, q1_root, multi_root, device):
    first = load_checkpoint(
        checkpoint_for(1, seeds[0], shot, q1_root, multi_root), device="cpu"
    )
    reference_state = encoder_state_from_checkpoint(first)
    encoder = (
        build_frozen_encoder(BACKBONE, pretrained=False, state_dict=reference_state)
        .to(device)
        .eval()
    )
    heads = {}
    for q in q_values:
        for seed in seeds:
            path = checkpoint_for(q, seed, shot, q1_root, multi_root)
            ckpt = load_checkpoint(path, device="cpu")
            stored_q = int(ckpt.get("query_group_size", 1))
            if stored_q != q:
                raise RuntimeError(f"{path}: query_group_size={stored_q}, expected {q}")
            model = WIPT(
                backbone=BACKBONE,
                pretrained=False,
                num_layers=int(ckpt.get("num_layers", 2)),
                num_heads=WIPT_HEADS,
                dropout=WIPT_DROPOUT,
                query_group_size=q,
            )
            load_wipt_state_compat(model, ckpt)
            verify_encoder(model, reference_state)
            model.encoder = nn.Identity()
            heads[(q, seed)] = model.to(device).eval()
    return encoder, heads


def summarize_runs(
    episodes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs = []
    for (dataset, q, seed), group in episodes.groupby(
        ["dataset", "query_group_size", "training_seed"], sort=False
    ):
        mean, ci = mean_ci95(group.accuracy)
        runs.append(
            {
                "dataset": dataset,
                "query_group_size": q,
                "training_seed": seed,
                "accuracy": mean,
                "episode_ci95": ci,
                "n_episodes": len(group),
            }
        )
    runs = pd.DataFrame(runs)

    summary = []
    for (dataset, q), group in runs.groupby(
        ["dataset", "query_group_size"], sort=False
    ):
        mean, sd, ci = t_ci95(group.accuracy)
        summary.append(
            {
                "dataset": dataset,
                "query_group_size": q,
                "mean_accuracy": mean,
                "training_seed_sd": sd,
                "training_seed_ci95": ci,
                "n_training_seeds": len(group),
                "mean_best_epoch": np.nan,
            }
        )
    summary = pd.DataFrame(summary)

    base = runs[runs.query_group_size == 1][
        ["dataset", "training_seed", "accuracy"]
    ].rename(columns={"accuracy": "q1_accuracy"})
    diffs = runs.merge(base, on=["dataset", "training_seed"], how="left")
    diffs["difference_vs_q1_pp"] = diffs.accuracy - diffs.q1_accuracy
    diff_summary = []
    for (dataset, q), group in diffs[diffs.query_group_size != 1].groupby(
        ["dataset", "query_group_size"], sort=False
    ):
        mean, sd, ci = t_ci95(group.difference_vs_q1_pp)
        diff_summary.append(
            {
                "dataset": dataset,
                "query_group_size": q,
                "mean_difference_vs_q1_pp": mean,
                "training_seed_sd_pp": sd,
                "training_seed_ci95_pp": ci,
                "positive_runs": int((group.difference_vs_q1_pp > 0).sum()),
                "negative_runs": int((group.difference_vs_q1_pp < 0).sum()),
            }
        )
    return runs, summary, pd.DataFrame(diff_summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot", type=int, default=5, choices=[1, 5])
    parser.add_argument("--query-groups", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--seeds", type=int, nargs="*", default=list(TRAIN_SEEDS))
    parser.add_argument(
        "--q1-checkpoint-root", type=Path, default=Path("checkpoints/seed_study")
    )
    parser.add_argument(
        "--multi-checkpoint-root", type=Path, default=Path("checkpoints/multiquery")
    )
    parser.add_argument("--out-root", type=Path, default=Path("results/raw/multiquery"))
    parser.add_argument(
        "--episodes-per-seed", type=int, default=CROSS_DOMAIN_EPISODES_PER_SEED
    )
    parser.add_argument("--encoder-batch-size", type=int, default=64)
    parser.add_argument("--cub", default=DEFAULT_DATA_PATHS["CUB"])
    parser.add_argument("--eurosat", default=DEFAULT_DATA_PATHS["EuroSAT"])
    parser.add_argument("--isic", default=DEFAULT_DATA_PATHS["ISIC"])
    parser.add_argument(
        "--extra-dataset",
        action="append",
        default=[],
        help="NAME=PATH; may be repeated",
    )
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    q_values = tuple(sorted(set(args.query_groups)))
    if 1 not in q_values:
        raise ValueError("Include q=1 so every q>1 condition has a matched reference.")
    device = get_device()
    encoder, heads = load_heads(
        q_values,
        seeds,
        args.shot,
        args.q1_checkpoint_root,
        args.multi_checkpoint_root,
        device,
    )

    datasets = {"CUB": args.cub, "EuroSAT": args.eurosat, "ISIC": args.isic}
    for item in args.extra_dataset:
        if "=" not in item:
            raise ValueError("--extra-dataset must be NAME=PATH")
        name, path = item.split("=", 1)
        datasets[name] = path

    rows = []
    group_diagnostics = []
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
                    shuffle_queries=True,
                )
                for episode_in_seed, (support, _, query, labels) in enumerate(loader):
                    support = support.to(device)
                    query = query.to(device)
                    labels = labels.to(device)
                    s = encode_in_chunks(
                        encoder, support, chunk_size=args.encoder_batch_size
                    )
                    qemb = encode_in_chunks(
                        encoder, query, chunk_size=args.encoder_batch_size
                    )
                    # Record query-group class diversity only as a post-hoc leakage check;
                    # labels are never passed to the WIPT grouping mechanism.
                    for qsize in q_values:
                        if qsize > 1:
                            diversities = []
                            homogeneous = []
                            for start in range(0, labels.numel(), qsize):
                                group_labels = labels[start : start + qsize]
                                diversities.append(
                                    int(torch.unique(group_labels).numel())
                                )
                                homogeneous.append(
                                    int(torch.unique(group_labels).numel() == 1)
                                )
                            group_diagnostics.append(
                                {
                                    "dataset": dataset,
                                    "query_group_size": qsize,
                                    "eval_seed": eval_seed,
                                    "episode_in_seed": episode_in_seed,
                                    "mean_classes_per_group": float(
                                        np.mean(diversities)
                                    ),
                                    "homogeneous_group_fraction": float(
                                        np.mean(homogeneous)
                                    ),
                                }
                            )
                        for seed in seeds:
                            scores, _, _, _ = heads[
                                (qsize, seed)
                            ].forward_from_embeddings(s, qemb, N_WAY, args.shot)
                            rows.append(
                                {
                                    "dataset": dataset,
                                    "query_group_size": qsize,
                                    "training_seed": seed,
                                    "eval_seed": eval_seed,
                                    "episode_in_seed": episode_in_seed,
                                    "episode": global_episode,
                                    "accuracy": 100.0
                                    * float(
                                        (scores.argmax(1) == labels).float().mean()
                                    ),
                                }
                            )
                    global_episode += 1
            print(f"{dataset}: q=1..{max(q_values)} evaluation complete")

    frame = pd.DataFrame(rows)
    runs, summary, diffs = summarize_runs(frame)
    out = args.out_root / shot_tag(args.shot) / "clean"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "episode_accuracy.csv", index=False)
    runs.to_csv(out / "run_accuracy.csv", index=False)
    summary.to_csv(out / "query_group_summary.csv", index=False)
    diffs.to_csv(out / "difference_vs_q1_summary.csv", index=False)
    if group_diagnostics:
        pd.DataFrame(group_diagnostics).to_csv(
            out / "query_group_leakage_check.csv", index=False
        )
    print(f"saved multi-query evaluation to {out}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
