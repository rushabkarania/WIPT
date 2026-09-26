"""Evaluate MAML-inspired support-supervised test-time adaptation on target domains."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
from torch.func import functional_call

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
    load_seed_study_heads,
    parse_seeds,
    protonet_scores,
    shot_tag,
    t_ci95,
)
from experimental.meta_heads import MetaEpisodicHead
from train.domain_shift import apply_shift, sample_shift
from utils.checkpoints import load_checkpoint
from utils.metrics import mean_ci95


def copy_standard_head(model, adaptation: str) -> MetaEpisodicHead:
    head = MetaEpisodicHead(adaptation=adaptation)
    source = model.state_dict()
    mapped = {}
    for key, value in source.items():
        if key.startswith("transformer_layers."):
            mapped["layers." + key[len("transformer_layers.") :]] = value
        elif key.startswith("norm."):
            mapped[key] = value
    incompatible = head.load_state_dict(mapped, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Could not map standard {adaptation} head: {incompatible}")
    return head


def adapted_scores(head, support, support_aug, support_labels, query, n_shot, inner_lr):
    params = dict(head.named_parameters())
    inner = nn.functional.cross_entropy(
        head(support, support_aug, N_WAY, n_shot), support_labels
    )
    grads = torch.autograd.grad(inner, tuple(params.values()), create_graph=False)
    adapted = {
        name: p - inner_lr * g.detach() for (name, p), g in zip(params.items(), grads)
    }
    return functional_call(head, adapted, (support, query, N_WAY, n_shot))


def load_meta_heads(root: Path, shot: int, seeds, device, reference_encoder_state):
    heads = {"joint": {}, "support": {}}
    for adaptation in heads:
        for seed in seeds:
            path = root / shot_tag(shot) / adaptation / f"seed_{seed}" / "model.pth"
            ckpt = load_checkpoint(path, "cpu")
            if int(ckpt["n_shot"]) != shot or int(ckpt["training_seed"]) != seed:
                raise RuntimeError(f"Metadata mismatch in {path}")
            enc = ckpt["encoder"]
            if set(enc) != set(reference_encoder_state) or any(
                not torch.equal(enc[k].cpu(), reference_encoder_state[k].cpu())
                for k in enc
            ):
                raise RuntimeError(
                    f"Frozen encoder in {path} differs from the reference seed study encoder."
                )
            head = MetaEpisodicHead(adaptation=adaptation)
            head.load_state_dict(ckpt["head"], strict=True)
            heads[adaptation][seed] = (
                head.to(device).eval(),
                float(ckpt["inner_lr"]),
                float(ckpt["support_aug_strength"]),
            )
    return heads


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shot", type=int, required=True, choices=[1, 5])
    p.add_argument("--seeds", type=int, nargs="*", default=list(TRAIN_SEEDS))
    p.add_argument(
        "--standard-checkpoint-root", type=Path, default=Path("checkpoints/seed_study")
    )
    p.add_argument(
        "--meta-checkpoint-root", type=Path, default=Path("checkpoints/meta_shift")
    )
    p.add_argument("--out-root", type=Path, default=Path("results/raw/meta_shift"))
    p.add_argument(
        "--episodes-per-seed", type=int, default=CROSS_DOMAIN_EPISODES_PER_SEED
    )
    p.add_argument("--encoder-batch-size", type=int, default=64)
    p.add_argument(
        "--test-inner-lr",
        type=float,
        default=None,
        help="Override checkpoint inner LR for all adapted conditions",
    )
    p.add_argument("--cub", default=DEFAULT_DATA_PATHS["CUB"])
    p.add_argument("--eurosat", default=DEFAULT_DATA_PATHS["EuroSAT"])
    p.add_argument("--isic", default=DEFAULT_DATA_PATHS["ISIC"])
    p.add_argument("--extra-dataset", action="append", default=[], help="NAME=PATH")
    args = p.parse_args()

    seeds = parse_seeds(args.seeds)
    device = get_device()
    encoder, standard_wipt, standard_support = load_seed_study_heads(
        args.standard_checkpoint_root,
        n_shot=args.shot,
        training_seeds=seeds,
        device=device,
    )
    reference_state = {k: v.detach().cpu() for k, v in encoder.state_dict().items()}
    meta = load_meta_heads(
        args.meta_checkpoint_root, args.shot, seeds, device, reference_state
    )

    standard_meta = {"joint": {}, "support": {}}
    for seed in seeds:
        standard_meta["joint"][seed] = (
            copy_standard_head(standard_wipt[seed], "joint").to(device).eval()
        )
        standard_meta["support"][seed] = (
            copy_standard_head(standard_support[seed], "support").to(device).eval()
        )

    datasets = {"CUB": args.cub, "EuroSAT": args.eurosat, "ISIC": args.isic}
    for item in args.extra_dataset:
        if "=" not in item:
            raise ValueError("--extra-dataset must be NAME=PATH")
        name, path = item.split("=", 1)
        datasets[name] = path

    rows = []
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
            for episode_in_seed, (
                support_images,
                support_labels,
                query_images,
                labels,
            ) in enumerate(loader):
                support_images = support_images.to(device)
                support_labels = support_labels.to(device)
                query_images = query_images.to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    support = encode_in_chunks(
                        encoder, support_images, chunk_size=args.encoder_batch_size
                    )
                    query = encode_in_chunks(
                        encoder, query_images, chunk_size=args.encoder_batch_size
                    )
                    pn = protonet_scores(support, query, N_WAY, args.shot)
                rows.append(
                    {
                        "dataset": dataset,
                        "model": "ProtoNet",
                        "training_seed": "fixed",
                        "eval_seed": eval_seed,
                        "episode": global_episode,
                        "accuracy": 100
                        * float((pn.argmax(1) == labels).float().mean()),
                    }
                )

                # One deterministic support augmentation is shared by every model/seed for this episode.
                with torch.random.fork_rng(
                    devices=[device] if device.type == "cuda" else []
                ):
                    aug_seed = 10_000_000 + int(eval_seed) * 10_000 + episode_in_seed
                    torch.manual_seed(aug_seed)
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(aug_seed)
                    # Use the mean training augmentation strength across meta checkpoints; they are expected equal.
                    representative_strength = next(iter(meta["joint"].values()))[2]
                    support_aug_images = apply_shift(
                        support_images, sample_shift(representative_strength)
                    )
                    with torch.no_grad():
                        support_aug = encode_in_chunks(
                            encoder,
                            support_aug_images,
                            chunk_size=args.encoder_batch_size,
                        )

                for seed in seeds:
                    with torch.no_grad():
                        sw, _, _, _ = standard_wipt[seed].forward_from_embeddings(
                            support, query, N_WAY, args.shot
                        )
                        ss = standard_support[seed].forward_from_embeddings(
                            support, query, N_WAY, args.shot
                        )
                    rows.extend(
                        [
                            {
                                "dataset": dataset,
                                "model": "Standard WIPT",
                                "training_seed": seed,
                                "eval_seed": eval_seed,
                                "episode": global_episode,
                                "accuracy": 100
                                * float((sw.argmax(1) == labels).float().mean()),
                            },
                            {
                                "dataset": dataset,
                                "model": "Standard support-only",
                                "training_seed": seed,
                                "eval_seed": eval_seed,
                                "episode": global_episode,
                                "accuracy": 100
                                * float((ss.argmax(1) == labels).float().mean()),
                            },
                        ]
                    )

                    # Naive TTA tests whether the adaptation step alone is enough without meta-training.
                    naive_lr = (
                        args.test_inner_lr
                        if args.test_inner_lr is not None
                        else meta["joint"][seed][1]
                    )
                    with torch.enable_grad():
                        nw = adapted_scores(
                            standard_meta["joint"][seed],
                            support,
                            support_aug,
                            support_labels,
                            query,
                            args.shot,
                            naive_lr,
                        )
                        ns = adapted_scores(
                            standard_meta["support"][seed],
                            support,
                            support_aug,
                            support_labels,
                            query,
                            args.shot,
                            naive_lr,
                        )
                    rows.extend(
                        [
                            {
                                "dataset": dataset,
                                "model": "Standard WIPT + naive TTA",
                                "training_seed": seed,
                                "eval_seed": eval_seed,
                                "episode": global_episode,
                                "accuracy": 100
                                * float(
                                    (nw.detach().argmax(1) == labels).float().mean()
                                ),
                            },
                            {
                                "dataset": dataset,
                                "model": "Standard support-only + naive TTA",
                                "training_seed": seed,
                                "eval_seed": eval_seed,
                                "episode": global_episode,
                                "accuracy": 100
                                * float(
                                    (ns.detach().argmax(1) == labels).float().mean()
                                ),
                            },
                        ]
                    )

                    for adaptation, label in [
                        ("joint", "MetaShift WIPT"),
                        ("support", "MetaShift support-only"),
                    ]:
                        head, train_lr, _ = meta[adaptation][seed]
                        with torch.no_grad():
                            base = head(support, query, N_WAY, args.shot)
                        lr = (
                            args.test_inner_lr
                            if args.test_inner_lr is not None
                            else train_lr
                        )
                        with torch.enable_grad():
                            adapted = adapted_scores(
                                head,
                                support,
                                support_aug,
                                support_labels,
                                query,
                                args.shot,
                                lr,
                            )
                        rows.extend(
                            [
                                {
                                    "dataset": dataset,
                                    "model": label + " (no TTA)",
                                    "training_seed": seed,
                                    "eval_seed": eval_seed,
                                    "episode": global_episode,
                                    "accuracy": 100
                                    * float((base.argmax(1) == labels).float().mean()),
                                },
                                {
                                    "dataset": dataset,
                                    "model": label + " + TTA",
                                    "training_seed": seed,
                                    "eval_seed": eval_seed,
                                    "episode": global_episode,
                                    "accuracy": 100
                                    * float(
                                        (adapted.detach().argmax(1) == labels)
                                        .float()
                                        .mean()
                                    ),
                                },
                            ]
                        )
                global_episode += 1
        print(f"{dataset}: meta-shift evaluation complete")

    episodes = pd.DataFrame(rows)
    run_rows = []
    for (dataset, model, seed), g in episodes.groupby(
        ["dataset", "model", "training_seed"], sort=False, dropna=False
    ):
        mean, ci = mean_ci95(g.accuracy)
        run_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "training_seed": seed,
                "accuracy": mean,
                "episode_ci95": ci,
                "n_episodes": len(g),
            }
        )
    runs = pd.DataFrame(run_rows)
    summary = []
    for (dataset, model), g in runs.groupby(["dataset", "model"], sort=False):
        if set(g.training_seed.astype(str)) == {"fixed"}:
            summary.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "mean_accuracy": float(g.accuracy.iloc[0]),
                    "training_seed_sd": np.nan,
                    "training_seed_ci95": np.nan,
                    "n_training_seeds": 0,
                }
            )
        else:
            mean, sd, ci = t_ci95(g.accuracy)
            summary.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "mean_accuracy": mean,
                    "training_seed_sd": sd,
                    "training_seed_ci95": ci,
                    "n_training_seeds": len(g),
                }
            )
    out = args.out_root / shot_tag(args.shot) / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(out / "episode_accuracy.csv", index=False)
    runs.to_csv(out / "run_accuracy.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "model_summary.csv", index=False)
    print(f"saved meta-shift evaluation to {out}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
