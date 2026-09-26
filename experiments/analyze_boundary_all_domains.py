"""Replicate boundary-local WIPT behaviour across target domains and shot regimes.

This is an evaluation-only experiment: it reuses the existing independent
seed-study checkpoints and never updates model parameters.

Outputs are written under::

    results/raw/boundary_all_domains/{shot}shot/{dataset}/

The script is restart-safe at the evaluation-seed level. Each completed
evaluation seed is cached under ``partials/`` and reused unless ``--overwrite``
is supplied.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

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
    t_ci95,
)

DATASET_CHOICES = ("CUB", "EuroSAT", "ISIC")
FINAL_FILES = (
    "queries_across_training_seeds.csv",
    "transitions_by_training_seed.csv",
    "margin_bins_by_training_seed.csv",
    "margin_bin_summary.csv",
    "transition_summary.csv",
)


def _parse_unique_ints(values: Iterable[int], *, name: str) -> tuple[int, ...]:
    parsed = tuple(int(v) for v in values)
    if not parsed:
        raise ValueError(f"At least one {name} is required.")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{name} values must be unique.")
    return parsed


def _shot_dir(root: Path, shot: int, dataset: str) -> Path:
    return root / f"{shot}shot" / dataset


def _is_complete(output: Path) -> bool:
    return all((output / name).exists() for name in FINAL_FILES)


def _safe_rate(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / max(denominator, 1)


def _analyse_queries(
    queries: pd.DataFrame,
    *,
    dataset: str,
    shot: int,
    training_seeds: tuple[int, ...],
    bins_requested: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # qcut can drop bins when many margins are tied. Record the actual number.
    queries = queries.copy()
    queries["margin_bin"] = (
        pd.qcut(
            queries.protonet_margin,
            bins_requested,
            labels=False,
            duplicates="drop",
        )
        + 1
    )

    transition_rows: list[dict] = []
    bin_rows: list[dict] = []

    proto_wrong = queries.protonet_correct == 0
    proto_right = ~proto_wrong
    n_errors = int(proto_wrong.sum())
    n_successes = int(proto_right.sum())

    for training_seed in training_seeds:
        column = f"wipt_correct_seed_{training_seed}"
        if column not in queries.columns:
            raise RuntimeError(f"Missing expected column: {column}")

        rescued = int((proto_wrong & (queries[column] == 1)).sum())
        broken = int((proto_right & (queries[column] == 0)).sum())
        wipt_accuracy = 100.0 * float(queries[column].mean())
        protonet_accuracy = 100.0 * float(queries.protonet_correct.mean())

        transition_rows.append(
            {
                "dataset": dataset,
                "shot": shot,
                "training_seed": training_seed,
                "n_queries": len(queries),
                "protonet_errors": n_errors,
                "protonet_successes": n_successes,
                "protonet_accuracy": protonet_accuracy,
                "wipt_accuracy": wipt_accuracy,
                "wipt_minus_protonet_pp": wipt_accuracy - protonet_accuracy,
                "rescued": rescued,
                "broken": broken,
                "rescue_rate_percent": _safe_rate(rescued, n_errors),
                "break_rate_percent": _safe_rate(broken, n_successes),
                "net_correct_change": rescued - broken,
            }
        )

        for margin_bin, group in queries.groupby("margin_bin", sort=True):
            diff = 100.0 * (
                group[column].astype(float) - group.protonet_correct.astype(float)
            )
            bin_rows.append(
                {
                    "dataset": dataset,
                    "shot": shot,
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

    summary_rows: list[dict] = []
    for margin_bin, group in bins.groupby("margin_bin", sort=True):
        mean, sd, ci = t_ci95(group.wipt_minus_protonet_pp)
        summary_rows.append(
            {
                "dataset": dataset,
                "shot": shot,
                "margin_bin": int(margin_bin),
                "n_queries_per_training_seed": int(group.n_queries.iloc[0]),
                "median_protonet_margin": float(group.median_protonet_margin.iloc[0]),
                "protonet_accuracy": float(group.protonet_accuracy.iloc[0]),
                "mean_wipt_accuracy": float(group.wipt_accuracy.mean()),
                "mean_wipt_minus_protonet_pp": mean,
                "training_seed_sd_pp": sd,
                "training_seed_ci95_pp": ci,
                "n_training_seeds": len(group),
                "positive_runs": int((group.wipt_minus_protonet_pp > 0).sum()),
                "negative_runs": int((group.wipt_minus_protonet_pp < 0).sum()),
                "zero_runs": int((group.wipt_minus_protonet_pp == 0).sum()),
            }
        )
    margin_summary = pd.DataFrame(summary_rows)

    transition_summary_rows: list[dict] = []
    for metric in (
        "wipt_minus_protonet_pp",
        "rescue_rate_percent",
        "break_rate_percent",
        "net_correct_change",
    ):
        mean, sd, ci = t_ci95(transitions[metric])
        transition_summary_rows.append(
            {
                "dataset": dataset,
                "shot": shot,
                "metric": metric,
                "mean": mean,
                "training_seed_sd": sd,
                "training_seed_ci95": ci,
                "n_training_seeds": len(transitions),
                "positive_runs": int((transitions[metric] > 0).sum()),
                "negative_runs": int((transitions[metric] < 0).sum()),
            }
        )
    transition_summary = pd.DataFrame(transition_summary_rows)

    return transitions, bins, margin_summary, transition_summary


def _evaluate_one_dataset(
    *,
    dataset: str,
    data_path: str,
    shot: int,
    training_seeds: tuple[int, ...],
    eval_seeds: tuple[int, ...],
    encoder,
    wipt_heads,
    device: torch.device,
    episodes_per_seed: int,
    bins_requested: int,
    encoder_batch_size: int,
    output: Path,
    overwrite: bool,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    partial_dir = output / "partials"

    if overwrite and output.exists():
        # Keep the output directory itself but clear cached/final files.
        if partial_dir.exists():
            shutil.rmtree(partial_dir)
        for name in FINAL_FILES:
            path = output / name
            if path.exists():
                path.unlink()

    if _is_complete(output) and not overwrite:
        print(f"skip complete {shot}-shot {dataset}: {output}")
        return

    partial_dir.mkdir(parents=True, exist_ok=True)
    queries_per_episode = N_WAY * N_QUERY

    with torch.no_grad():
        for eval_index, eval_seed in enumerate(eval_seeds):
            partial_path = partial_dir / f"eval_seed_{eval_seed}.csv"
            if partial_path.exists() and not overwrite:
                print(f"reuse {shot}-shot {dataset} eval seed {eval_seed}")
                continue

            rows: list[dict] = []
            loader = episode_loader(
                data_path,
                eval_seed,
                episodes_per_seed,
                n_way=N_WAY,
                n_shot=shot,
                n_query=N_QUERY,
            )

            for episode, (support, _, query, labels) in enumerate(loader):
                support = support.to(device)
                query = query.to(device)
                labels = labels.to(device)

                support_embeddings = encode_in_chunks(
                    encoder, support, chunk_size=encoder_batch_size
                )
                query_embeddings = encode_in_chunks(
                    encoder, query, chunk_size=encoder_batch_size
                )

                proto_scores = protonet_scores(
                    support_embeddings,
                    query_embeddings,
                    N_WAY,
                    shot,
                )
                proto_pred = proto_scores.argmax(dim=1)
                proto_margin = query_margin(proto_scores, labels)

                wipt_correct: dict[int, torch.Tensor] = {}
                for training_seed in training_seeds:
                    scores, _, _, _ = wipt_heads[training_seed].forward_from_embeddings(
                        support_embeddings,
                        query_embeddings,
                        N_WAY,
                        shot,
                    )
                    wipt_correct[training_seed] = (
                        (scores.argmax(dim=1) == labels).int().cpu()
                    )

                proto_correct = (proto_pred == labels).int().cpu()
                margins = proto_margin.cpu()
                labels_cpu = labels.cpu()

                base_query_id = (
                    eval_index * episodes_per_seed * queries_per_episode
                    + episode * queries_per_episode
                )
                for index in range(labels.shape[0]):
                    row = {
                        "dataset": dataset,
                        "shot": shot,
                        "query_id": base_query_id + index,
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

            pd.DataFrame(rows).to_csv(partial_path, index=False)
            print(
                f"{shot}-shot {dataset} eval seed {eval_seed}: "
                f"boundary analysis complete ({len(rows):,} queries)"
            )

    partial_frames = []
    for eval_seed in eval_seeds:
        partial_path = partial_dir / f"eval_seed_{eval_seed}.csv"
        if not partial_path.exists():
            raise RuntimeError(f"Missing partial result: {partial_path}")
        partial_frames.append(pd.read_csv(partial_path))

    queries = pd.concat(partial_frames, ignore_index=True)
    transitions, bins, margin_summary, transition_summary = _analyse_queries(
        queries,
        dataset=dataset,
        shot=shot,
        training_seeds=training_seeds,
        bins_requested=bins_requested,
    )

    queries.to_csv(output / "queries_across_training_seeds.csv", index=False)
    transitions.to_csv(output / "transitions_by_training_seed.csv", index=False)
    bins.to_csv(output / "margin_bins_by_training_seed.csv", index=False)
    margin_summary.to_csv(output / "margin_bin_summary.csv", index=False)
    transition_summary.to_csv(output / "transition_summary.csv", index=False)

    print(f"saved {shot}-shot {dataset} boundary outputs to {output}")


def _write_combined_summaries(
    *,
    out_root: Path,
    shots: tuple[int, ...],
    datasets: tuple[str, ...],
) -> None:
    transition_frames = []
    margin_frames = []
    for shot in shots:
        for dataset in datasets:
            output = _shot_dir(out_root, shot, dataset)
            t_path = output / "transition_summary.csv"
            m_path = output / "margin_bin_summary.csv"
            if t_path.exists():
                transition_frames.append(pd.read_csv(t_path))
            if m_path.exists():
                margin_frames.append(pd.read_csv(m_path))

    if transition_frames:
        pd.concat(transition_frames, ignore_index=True).to_csv(
            out_root / "all_domain_transition_summary.csv", index=False
        )
    if margin_frames:
        pd.concat(margin_frames, ignore_index=True).to_csv(
            out_root / "all_domain_margin_bin_summary.csv", index=False
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replicate ProtoNet-boundary WIPT behaviour across target domains "
            "and 1/5-shot regimes using existing seed-study checkpoints."
        )
    )
    parser.add_argument(
        "--shots",
        type=int,
        nargs="+",
        default=[1, 5],
        choices=[1, 5],
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_CHOICES),
        choices=list(DATASET_CHOICES),
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=list(TRAIN_SEEDS))
    parser.add_argument("--eval-seeds", type=int, nargs="*", default=list(EVAL_SEEDS))
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("checkpoints/seed_study"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("results/raw/boundary_all_domains"),
    )
    parser.add_argument(
        "--episodes-per-seed",
        type=int,
        default=CROSS_DOMAIN_EPISODES_PER_SEED,
    )
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--encoder-batch-size", type=int, default=64)
    parser.add_argument("--cub", default=DEFAULT_DATA_PATHS["CUB"])
    parser.add_argument("--eurosat", default=DEFAULT_DATA_PATHS["EuroSAT"])
    parser.add_argument("--isic", default=DEFAULT_DATA_PATHS["ISIC"])
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even when cached/final outputs already exist.",
    )
    args = parser.parse_args()

    if args.episodes_per_seed <= 0:
        raise ValueError("--episodes-per-seed must be positive")
    if args.bins < 2:
        raise ValueError("--bins must be at least 2")
    if args.encoder_batch_size <= 0:
        raise ValueError("--encoder-batch-size must be positive")

    shots = _parse_unique_ints(args.shots, name="shot")
    training_seeds = parse_seeds(args.seeds)
    eval_seeds = _parse_unique_ints(args.eval_seeds, name="evaluation seed")
    datasets = tuple(args.datasets)
    if len(set(datasets)) != len(datasets):
        raise ValueError("Dataset names must be unique.")

    paths = {
        "CUB": args.cub,
        "EuroSAT": args.eurosat,
        "ISIC": args.isic,
    }

    device = get_device()
    print(f"device={device}")
    print(f"shots={shots} datasets={datasets}")
    print(f"training_seeds={training_seeds} eval_seeds={eval_seeds}")

    args.out_root.mkdir(parents=True, exist_ok=True)

    for shot in shots:
        # load_seed_study_heads also loads the support-only controls, which are
        # not used here, but reusing this helper preserves the exact
        # same frozen encoder/reference state and checkpoint validation as the
        # main seed-study evaluation.
        encoder, wipt_heads, _ = load_seed_study_heads(
            args.checkpoint_root,
            n_shot=shot,
            training_seeds=training_seeds,
            device=device,
        )

        for dataset in datasets:
            output = _shot_dir(args.out_root, shot, dataset)
            _evaluate_one_dataset(
                dataset=dataset,
                data_path=paths[dataset],
                shot=shot,
                training_seeds=training_seeds,
                eval_seeds=eval_seeds,
                encoder=encoder,
                wipt_heads=wipt_heads,
                device=device,
                episodes_per_seed=args.episodes_per_seed,
                bins_requested=args.bins,
                encoder_batch_size=args.encoder_batch_size,
                output=output,
                overwrite=args.overwrite,
            )

        # Explicitly release per-shot models before loading the next shot.
        del encoder, wipt_heads
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_combined_summaries(
        out_root=args.out_root,
        shots=shots,
        datasets=datasets,
    )
    print(f"combined summaries saved under {args.out_root}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
