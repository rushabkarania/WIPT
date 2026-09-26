"""Train WIPT/support-only heads with explicit source pseudo-domain shifts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from configs.experiment import (
    BACKBONE,
    DEFAULT_DATA_PATHS,
    EPOCHS,
    ETA_MIN,
    LEARNING_RATE,
    N_SHOT,
    N_WAY,
    WIPT_DROPOUT,
    WIPT_HEADS,
    NUM_WORKERS,
)
from models.baselines import SupportTransformerViT
from models.wipt import WIPT
from train.domain_shift import shift_episode
from train.training_utils import evaluate, make_train_val_loaders
from utils.metrics import accuracy, mean_ci95, set_seed


def evaluate_shifted(
    model, loader, device, *, n_shot: int, mode: str, strength: float, seed: int
):
    model.eval()
    clean_values, shifted_values = [], []
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            for support, _, query, labels in loader:
                support, query, labels = (
                    support.to(device),
                    query.to(device),
                    labels.to(device),
                )
                clean_values.append(
                    accuracy(model(support, query, N_WAY, n_shot), labels)
                )
                ss, sq, _ = shift_episode(support, query, mode=mode, strength=strength)
                shifted_values.append(accuracy(model(ss, sq, N_WAY, n_shot), labels))
    clean_mean, clean_ci = mean_ci95(clean_values)
    shift_mean, shift_ci = mean_ci95(shifted_values)
    return clean_mean, clean_ci, shift_mean, shift_ci


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["wipt", "support"])
    parser.add_argument(
        "--mode", choices=["shared", "cross", "mixed"], default="shared"
    )
    parser.add_argument("--shift-weight", type=float, default=0.5)
    parser.add_argument("--max-strength", type=float, default=1.0)
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--train-dir", default=DEFAULT_DATA_PATHS["miniImageNet_train"])
    parser.add_argument("--val-dir", default=DEFAULT_DATA_PATHS["miniImageNet_val"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-shot", type=int, default=N_SHOT, choices=[1, 5])
    args = parser.parse_args()

    if not 0.0 <= args.shift_weight <= 1.0:
        raise ValueError("shift-weight must be in [0,1].")
    set_seed(args.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = make_train_val_loaders(
        args.train_dir, args.val_dir, args.workers, n_shot=args.n_shot, seed=args.seed
    )
    if args.model == "wipt":
        model = WIPT(
            backbone=BACKBONE,
            pretrained=True,
            num_layers=2,
            num_heads=WIPT_HEADS,
            dropout=WIPT_DROPOUT,
        ).to(device)
    else:
        model = SupportTransformerViT(
            backbone=BACKBONE,
            pretrained=True,
            num_layers=2,
            num_heads=WIPT_HEADS,
            dropout=WIPT_DROPOUT,
        ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable, lr=LEARNING_RATE, weight_decay=0.0)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=ETA_MIN)
    loss_fn = nn.CrossEntropyLoss()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    best_val = -float("inf")
    history = []
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        if args.no_curriculum:
            strength = args.max_strength
        else:
            progress = (epoch - 1) / max(EPOCHS - 1, 1)
            strength = args.max_strength * (0.25 + 0.75 * progress)
        losses, clean_accs, shift_accs = [], [], []
        mode_counts = {"shared": 0, "cross": 0}
        clean_episodes = 0
        shifted_episodes = 0
        # Stochastic mixture estimator of the weighted clean/shifted objective.
        # With probability shift_weight, train on a pseudo-domain episode; otherwise
        # train on the clean episode. This has the same expected objective as computing
        # both losses every iteration, but requires only one model/encoder forward pass.
        for support, _, query, labels in tqdm(
            train_loader, desc=f"shift-{args.mode}", leave=False
        ):
            support, query, labels = (
                support.to(device),
                query.to(device),
                labels.to(device),
            )
            optimizer.zero_grad(set_to_none=True)

            use_shift = bool(torch.rand((), device=device) < args.shift_weight)
            if use_shift:
                shifted_support, shifted_query, realized_mode = shift_episode(
                    support, query, mode=args.mode, strength=strength
                )
                mode_counts[realized_mode] += 1
                scores = model(shifted_support, shifted_query, N_WAY, args.n_shot)
                shift_accs.append(accuracy(scores, labels))
                shifted_episodes += 1
            else:
                scores = model(support, query, N_WAY, args.n_shot)
                clean_accs.append(accuracy(scores, labels))
                clean_episodes += 1

            loss = loss_fn(scores, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        val_acc, val_ci, shifted_val_acc, shifted_val_ci = evaluate_shifted(
            model,
            val_loader,
            device,
            n_shot=args.n_shot,
            mode=args.mode,
            strength=args.max_strength,
            seed=args.seed + 20_000,
        )
        selection_score = (
            1.0 - args.shift_weight
        ) * val_acc + args.shift_weight * shifted_val_acc
        scheduler.step()
        elapsed = time.time() - started
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "clean_train_accuracy": (
                float(np.mean(clean_accs)) if clean_accs else float("nan")
            ),
            "shifted_train_accuracy": (
                float(np.mean(shift_accs)) if shift_accs else float("nan")
            ),
            "clean_episodes": clean_episodes,
            "shifted_episodes": shifted_episodes,
            "validation_accuracy": val_acc,
            "validation_ci95": val_ci,
            "shifted_validation_accuracy": shifted_val_acc,
            "shifted_validation_ci95": shifted_val_ci,
            "selection_score": selection_score,
            "shift_strength": strength,
            "shared_episodes": mode_counts["shared"],
            "cross_episodes": mode_counts["cross"],
            "elapsed_seconds": elapsed,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{EPOCHS} | loss {row['loss']:.4f} | clean {100*row['clean_train_accuracy']:.2f}% "
            f"| shifted {100*row['shifted_train_accuracy']:.2f}% | clean val {100*val_acc:.2f}% "
            f"| shifted val {100*shifted_val_acc:.2f}% | select {100*selection_score:.2f}% | strength {strength:.2f}"
        )
        if selection_score > best_val:
            best_val = selection_score
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_acc": val_acc,
                    "val_ci": val_ci,
                    "shifted_val_acc": shifted_val_acc,
                    "shifted_val_ci": shifted_val_ci,
                    "selection_score": selection_score,
                    "elapsed_seconds_to_best": elapsed,
                    "backbone": BACKBONE,
                    "training_seed": args.seed,
                    "n_shot": args.n_shot,
                    "num_layers": 2,
                    "query_group_size": 1,
                    "shift_aware": True,
                    "shift_mode": args.mode,
                    "shift_weight": args.shift_weight,
                    "max_shift_strength": args.max_strength,
                    "model_family": args.model,
                },
                args.output,
            )
            print(f"saved {args.output}")

    history_path = args.history or args.output.with_name(
        args.output.stem + "_history.csv"
    )
    with Path(history_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
