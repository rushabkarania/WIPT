"""Parameter, complexity, latency, and backward-cost benchmark for episodic heads.

The default benchmark operates on frozen-encoder embeddings so it isolates the episodic
head.  This is the relevant difference because every controlled condition shares the same
ViT image encoder.  CUDA peak memory is reported when available.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import pandas as pd
import torch
import torch.nn as nn

from configs.experiment import EMBED_DIM, N_QUERY, N_WAY
from models.baselines import SupportTransformerViT
from models.factorial import FactorialFewShotModel
from models.wipt import WIPT


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(
    fn, *, device: torch.device, warmup: int, repeats: int
) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    sync(device)
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        sync(device)
        values.append(1000.0 * (time.perf_counter() - start))
    series = pd.Series(values)
    return float(series.median()), float(series.quantile(0.95))


def joint_token_pairs(
    n_support: int, n_queries: int, group_size: int, layers: int = 2
) -> int:
    total = 0
    for start in range(0, n_queries, group_size):
        q = min(group_size, n_queries - start)
        total += (n_support + q) ** 2
    return layers * total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot", type=int, default=5, choices=[1, 5])
    parser.add_argument("--n-query-per-class", type=int, default=N_QUERY)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output", type=Path, default=Path("results/raw/efficiency.csv")
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    n_support = N_WAY * args.shot
    n_queries = N_WAY * args.n_query_per_class
    support = torch.randn(n_support, EMBED_DIM, device=device)
    query = torch.randn(n_queries, EMBED_DIM, device=device)
    labels = torch.arange(N_WAY, device=device).repeat_interleave(
        args.n_query_per_class
    )
    loss_fn = nn.CrossEntropyLoss()

    rows = []

    # ProtoNet head has no trainable module and no attention.
    def proto_forward():
        prototypes = support.view(N_WAY, args.shot, -1).mean(1)
        return -torch.cdist(query, prototypes)

    fmed, fp95 = timed(
        proto_forward, device=device, warmup=args.warmup, repeats=args.repeats
    )
    rows.append(
        {
            "condition": "ProtoNet",
            "query_group_size": 0,
            "trainable_parameters": 0,
            "sequence_tokens": n_support,
            "attention_token_pairs_per_episode": 0,
            "forward_median_ms": fmed,
            "forward_p95_ms": fp95,
            "backward_median_ms": None,
            "backward_p95_ms": None,
            "peak_memory_mb": None,
        }
    )

    support_model = SupportTransformerViT(pretrained=False, num_layers=2).to(device)
    support_model.encoder = nn.Identity()
    support_model.eval()
    trainable = sum(p.numel() for p in support_model.parameters() if p.requires_grad)
    with torch.no_grad():
        fmed, fp95 = timed(
            lambda: support_model.forward_from_embeddings(
                support, query, N_WAY, args.shot
            ),
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
    rows.append(
        {
            "condition": "Support-only Transformer",
            "query_group_size": 0,
            "trainable_parameters": trainable,
            "sequence_tokens": n_support,
            "attention_token_pairs_per_episode": 2 * n_support**2,
            "forward_median_ms": fmed,
            "forward_p95_ms": fp95,
            "backward_median_ms": None,
            "backward_p95_ms": None,
            "peak_memory_mb": None,
        }
    )

    for group_size in [1, 2, 3, 4, 5]:
        model = WIPT(pretrained=False, num_layers=2, query_group_size=group_size).to(
            device
        )
        model.encoder = nn.Identity()
        model.eval()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            fmed, fp95 = timed(
                lambda: model.forward_from_embeddings(support, query, N_WAY, args.shot)[
                    0
                ],
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        peak = (
            (torch.cuda.max_memory_allocated(device) / 2**20)
            if device.type == "cuda"
            else None
        )

        model.train()

        def train_step():
            model.zero_grad(set_to_none=True)
            scores = model.forward_from_embeddings(support, query, N_WAY, args.shot)[0]
            loss = loss_fn(scores, labels)
            loss.backward()
            return loss

        bmed, bp95 = timed(
            train_step,
            device=device,
            warmup=max(3, args.warmup // 4),
            repeats=max(10, args.repeats // 4),
        )
        rows.append(
            {
                "condition": f"WIPT q={group_size}",
                "query_group_size": group_size,
                "trainable_parameters": trainable,
                "sequence_tokens": n_support + group_size,
                "attention_token_pairs_per_episode": joint_token_pairs(
                    n_support, n_queries, group_size
                ),
                "forward_median_ms": fmed,
                "forward_p95_ms": fp95,
                "backward_median_ms": bmed,
                "backward_p95_ms": bp95,
                "peak_memory_mb": peak,
            }
        )

    # Parameter counts for the full 3x3 matrix, useful for the manuscript table.
    for adaptation in ["none", "support", "joint"]:
        for scorer in ["euclidean", "cosine", "relation"]:
            model = FactorialFewShotModel(
                adaptation=adaptation, scorer=scorer, pretrained=False
            ).to(device)
            model.encoder = nn.Identity()
            rows.append(
                {
                    "condition": f"PARAMS: {adaptation}/{scorer}",
                    "query_group_size": 1 if adaptation == "joint" else 0,
                    "trainable_parameters": model.trainable_parameter_count,
                    "sequence_tokens": None,
                    "attention_token_pairs_per_episode": None,
                    "forward_median_ms": None,
                    "forward_p95_ms": None,
                    "backward_median_ms": None,
                    "backward_p95_ms": None,
                    "peak_memory_mb": None,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
