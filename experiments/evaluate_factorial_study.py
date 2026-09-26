"""Matched-episode evaluation for the 3 x 3 adaptation x decision-rule study."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from configs.experiment import (
    BACKBONE,
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
    parse_seeds,
    seed_checkpoint_dir,
    shot_tag,
    t_ci95,
)
from models.backbone import build_frozen_encoder
from models.factorial import (
    ADAPTATION_MODES,
    SCORERS,
    FactorialFewShotModel,
    condition_name,
)
from utils.checkpoints import (
    encoder_state_from_checkpoint,
    load_checkpoint,
    verify_encoder,
)
from utils.metrics import mean_ci95


def slug(adaptation: str, scorer: str) -> str:
    return f"{adaptation}_{scorer}"


def is_fixed(adaptation: str, scorer: str) -> bool:
    return adaptation == "none" and scorer in {"euclidean", "cosine"}


def seed_study_path(
    reference_root: Path, shot: int, adaptation: str, seed: int
) -> Path:
    base = reference_root / shot_tag(shot) / f"seed_{seed}"
    if adaptation == "support":
        return base / "support_transformer.pth"
    if adaptation == "joint":
        return base / "wipt_l2.pth"
    raise ValueError(adaptation)


def load_models(args, seeds, device):
    ref_path = (
        seed_checkpoint_dir(args.reference_checkpoint_root, args.shot, seeds[0])
        / "wipt_l2.pth"
    )
    ref_ckpt = load_checkpoint(ref_path, device="cpu")
    reference_state = encoder_state_from_checkpoint(ref_ckpt)
    encoder = (
        build_frozen_encoder(BACKBONE, pretrained=False, state_dict=reference_state)
        .to(device)
        .eval()
    )

    models = {}
    for adaptation in ADAPTATION_MODES:
        for scorer in SCORERS:
            key = (adaptation, scorer)
            if is_fixed(adaptation, scorer):
                model = FactorialFewShotModel(
                    adaptation=adaptation,
                    scorer=scorer,
                    backbone=BACKBONE,
                    pretrained=False,
                    encoder_state_dict=reference_state,
                )
                verify_encoder(model, reference_state)
                model.encoder = nn.Identity()
                models[key] = {"fixed": model.to(device).eval()}
                continue
            per_seed = {}
            for seed in seeds:
                if scorer == "euclidean" and adaptation in {"support", "joint"}:
                    path = seed_study_path(
                        args.reference_checkpoint_root, args.shot, adaptation, seed
                    )
                else:
                    path = (
                        args.checkpoint_root
                        / shot_tag(args.shot)
                        / slug(adaptation, scorer)
                        / f"seed_{seed}"
                        / "model.pth"
                    )
                ckpt = load_checkpoint(path, device="cpu")
                model = FactorialFewShotModel(
                    adaptation=adaptation,
                    scorer=scorer,
                    backbone=BACKBONE,
                    pretrained=False,
                    encoder_state_dict=reference_state,
                    query_group_size=int(ckpt.get("query_group_size", 1)),
                )
                incompatible = model.load_state_dict(ckpt["model"], strict=True)
                if incompatible.missing_keys or incompatible.unexpected_keys:
                    raise RuntimeError(f"{path}: state mismatch")
                verify_encoder(model, reference_state)
                model.encoder = nn.Identity()
                per_seed[seed] = model.to(device).eval()
            models[key] = per_seed
    return encoder, models


def summarize(episodes: pd.DataFrame):
    runs = []
    for keys, group in episodes.groupby(
        ["dataset", "adaptation", "scorer", "condition", "training_seed"],
        sort=False,
        dropna=False,
    ):
        dataset, adaptation, scorer, condition, seed = keys
        mean, ci = mean_ci95(group.accuracy)
        runs.append(
            {
                "dataset": dataset,
                "adaptation": adaptation,
                "scorer": scorer,
                "condition": condition,
                "training_seed": seed,
                "accuracy": mean,
                "episode_ci95": ci,
                "n_episodes": len(group),
            }
        )
    runs = pd.DataFrame(runs)

    summary = []
    for keys, group in runs.groupby(
        ["dataset", "adaptation", "scorer", "condition"], sort=False
    ):
        dataset, adaptation, scorer, condition = keys
        if set(group.training_seed.astype(str)) == {"fixed"}:
            value = float(group.accuracy.iloc[0])
            summary.append(
                {
                    "dataset": dataset,
                    "adaptation": adaptation,
                    "scorer": scorer,
                    "condition": condition,
                    "mean_accuracy": value,
                    "training_seed_sd": np.nan,
                    "training_seed_ci95": np.nan,
                    "n_training_seeds": 0,
                }
            )
        else:
            mean, sd, ci = t_ci95(group.accuracy)
            summary.append(
                {
                    "dataset": dataset,
                    "adaptation": adaptation,
                    "scorer": scorer,
                    "condition": condition,
                    "mean_accuracy": mean,
                    "training_seed_sd": sd,
                    "training_seed_ci95": ci,
                    "n_training_seeds": len(group),
                }
            )
    summary = pd.DataFrame(summary)

    comparisons = []
    for dataset in runs.dataset.unique():
        d = runs[runs.dataset == dataset]
        for scorer in SCORERS:
            ds = d[d.scorer == scorer]
            raw = ds[ds.adaptation == "none"]
            support = ds[ds.adaptation == "support"]
            joint = ds[ds.adaptation == "joint"]
            for lhs_name, lhs, rhs_name, rhs in [
                ("WIPT", joint, "Raw", raw),
                ("WIPT", joint, "Support-only", support),
                ("Support-only", support, "Raw", raw),
            ]:
                if lhs.empty or rhs.empty:
                    continue
                values = []
                for seed in sorted(set(lhs.training_seed) - {"fixed"}):
                    lv = float(lhs[lhs.training_seed == seed].accuracy.iloc[0])
                    if "fixed" in set(rhs.training_seed.astype(str)):
                        rv = float(rhs.accuracy.iloc[0])
                    else:
                        rv = float(rhs[rhs.training_seed == seed].accuracy.iloc[0])
                    values.append(lv - rv)
                mean, sd, ci = t_ci95(values)
                comparisons.append(
                    {
                        "dataset": dataset,
                        "scorer": scorer,
                        "comparison": f"{lhs_name} - {rhs_name}",
                        "mean_difference_pp": mean,
                        "training_seed_sd_pp": sd,
                        "training_seed_ci95_pp": ci,
                        "positive_runs": int(sum(v > 0 for v in values)),
                        "negative_runs": int(sum(v < 0 for v in values)),
                    }
                )
    return runs, summary, pd.DataFrame(comparisons)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot", type=int, default=5, choices=[1, 5])
    parser.add_argument("--seeds", type=int, nargs="*", default=list(TRAIN_SEEDS))
    parser.add_argument(
        "--reference-checkpoint-root", type=Path, default=Path("checkpoints/seed_study")
    )
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("checkpoints/factorial")
    )
    parser.add_argument("--out-root", type=Path, default=Path("results/raw/factorial"))
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
    encoder, models = load_models(args, seeds, device)
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
                    for adaptation in ADAPTATION_MODES:
                        for scorer in SCORERS:
                            for seed, model in models[(adaptation, scorer)].items():
                                scores = model.forward_from_embeddings(
                                    s, q, N_WAY, args.shot
                                )
                                rows.append(
                                    {
                                        "dataset": dataset,
                                        "adaptation": adaptation,
                                        "scorer": scorer,
                                        "condition": condition_name(adaptation, scorer),
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
            print(f"{dataset}: factorial evaluation complete")

    episodes = pd.DataFrame(rows)
    runs, summary, comparisons = summarize(episodes)
    out = args.out_root / shot_tag(args.shot) / "clean"
    out.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(out / "episode_accuracy.csv", index=False)
    runs.to_csv(out / "run_accuracy.csv", index=False)
    summary.to_csv(out / "condition_summary.csv", index=False)
    comparisons.to_csv(out / "adaptation_comparisons.csv", index=False)
    print(f"saved factorial evaluation to {out}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
