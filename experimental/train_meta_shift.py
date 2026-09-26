"""First-order MAML-inspired training for target-support test-time adaptation.

For each source episode we create a pseudo-target appearance domain.  An inner update
uses only the labelled support set: augmented views of those support images act as
adaptation queries while the original pseudo-target support images remain the context.
The outer objective evaluates real episode queries after that one-step update.  At test
time the same support-only adaptation operation can be applied on a novel target domain.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call
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
    NUM_WORKERS,
)
from models.backbone import build_frozen_encoder
from experimental.meta_heads import MetaEpisodicHead
from train.domain_shift import apply_shift, sample_shift
from train.training_utils import make_train_val_loaders
from utils.metrics import accuracy, mean_ci95, set_seed


def encode(encoder: nn.Module, images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return encoder(images)


def adapted_parameters(
    head: nn.Module, inner_loss: torch.Tensor, inner_lr: float
) -> dict[str, torch.Tensor]:
    params = dict(head.named_parameters())
    grads = torch.autograd.grad(inner_loss, tuple(params.values()), create_graph=False)
    # First-order MAML: gradient values are treated as constants in the outer derivative.
    return {
        name: param - inner_lr * grad.detach()
        for (name, param), grad in zip(params.items(), grads)
    }


def one_step_scores(
    head: MetaEpisodicHead,
    support_embeddings: torch.Tensor,
    support_aug_embeddings: torch.Tensor,
    support_labels: torch.Tensor,
    query_embeddings: torch.Tensor,
    *,
    n_way: int,
    n_shot: int,
    inner_lr: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    inner_scores = head(support_embeddings, support_aug_embeddings, n_way, n_shot)
    inner_loss = nn.functional.cross_entropy(inner_scores, support_labels)
    adapted = adapted_parameters(head, inner_loss, inner_lr)
    outer_scores = functional_call(
        head, adapted, (support_embeddings, query_embeddings, n_way, n_shot)
    )
    return outer_scores, inner_loss


def evaluate_adapted(
    head,
    encoder,
    loader,
    device,
    *,
    n_shot: int,
    inner_lr: float,
    aug_strength: float,
    seed: int,
):
    head.eval()
    encoder.eval()
    accuracies = []
    # Make the support augmentation repeatable across epochs without perturbing training RNG.
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        for support, support_labels, query, labels in loader:
            support = support.to(device)
            support_labels = support_labels.to(device)
            query = query.to(device)
            labels = labels.to(device)
            support_aug = apply_shift(support, sample_shift(aug_strength))
            s = encode(encoder, support)
            sa = encode(encoder, support_aug)
            q = encode(encoder, query)
            with torch.enable_grad():
                scores, _ = one_step_scores(
                    head,
                    s,
                    sa,
                    support_labels,
                    q,
                    n_way=N_WAY,
                    n_shot=n_shot,
                    inner_lr=inner_lr,
                )
            accuracies.append(accuracy(scores.detach(), labels))
    return mean_ci95(accuracies)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptation", choices=["joint", "support"], default="joint")
    parser.add_argument("--train-dir", default=DEFAULT_DATA_PATHS["miniImageNet_train"])
    parser.add_argument("--val-dir", default=DEFAULT_DATA_PATHS["miniImageNet_val"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-shot", type=int, default=N_SHOT, choices=[1, 5])
    parser.add_argument("--inner-lr", type=float, default=1e-3)
    parser.add_argument("--domain-strength", type=float, default=0.8)
    parser.add_argument("--support-aug-strength", type=float, default=0.45)
    parser.add_argument("--clean-outer-weight", type=float, default=0.25)
    args = parser.parse_args()

    set_seed(args.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = make_train_val_loaders(
        args.train_dir, args.val_dir, args.workers, n_shot=args.n_shot, seed=args.seed
    )
    encoder = build_frozen_encoder(BACKBONE, pretrained=True).to(device).eval()
    head = MetaEpisodicHead(args.adaptation).to(device)
    optimizer = Adam(head.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=ETA_MIN)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    best_val = -float("inf")
    history = []
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        head.train()
        losses, inner_losses, outer_accs = [], [], []
        for support, support_labels, query, labels in tqdm(
            train_loader, desc=f"meta-{args.adaptation}", leave=False
        ):
            support = support.to(device)
            support_labels = support_labels.to(device)
            query = query.to(device)
            labels = labels.to(device)

            # One episode-wide pseudo-target domain for both support and query.
            domain_params = sample_shift(args.domain_strength)
            target_support = apply_shift(support, domain_params)
            target_query = apply_shift(query, domain_params)
            # A second view of the labelled target support provides the inner-loop signal.
            support_aug = apply_shift(
                target_support, sample_shift(args.support_aug_strength)
            )

            s = encode(encoder, target_support)
            sa = encode(encoder, support_aug)
            q = encode(encoder, target_query)
            clean_s = encode(encoder, support)
            clean_q = encode(encoder, query)

            optimizer.zero_grad(set_to_none=True)
            outer_scores, inner_loss = one_step_scores(
                head,
                s,
                sa,
                support_labels,
                q,
                n_way=N_WAY,
                n_shot=args.n_shot,
                inner_lr=args.inner_lr,
            )
            shifted_outer = nn.functional.cross_entropy(outer_scores, labels)
            clean_scores = head(clean_s, clean_q, N_WAY, args.n_shot)
            clean_loss = nn.functional.cross_entropy(clean_scores, labels)
            loss = shifted_outer + args.clean_outer_weight * clean_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            inner_losses.append(float(inner_loss.item()))
            outer_accs.append(accuracy(outer_scores.detach(), labels))

        val_acc, val_ci = evaluate_adapted(
            head,
            encoder,
            val_loader,
            device,
            n_shot=args.n_shot,
            inner_lr=args.inner_lr,
            aug_strength=args.support_aug_strength,
            seed=args.seed + 30_000,
        )
        scheduler.step()
        elapsed = time.time() - started
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "inner_loss": float(np.mean(inner_losses)),
            "shifted_outer_train_accuracy": float(np.mean(outer_accs)),
            "adapted_validation_accuracy": val_acc,
            "adapted_validation_ci95": val_ci,
            "elapsed_seconds": elapsed,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{EPOCHS} | loss {row['loss']:.4f} | outer {100*row['shifted_outer_train_accuracy']:.2f}% | adapted val {100*val_acc:.2f}% ± {100*val_ci:.2f}%"
        )
        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "head": head.state_dict(),
                    "encoder": encoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_acc": best_val,
                    "val_ci": val_ci,
                    "elapsed_seconds_to_best": elapsed,
                    "backbone": BACKBONE,
                    "training_seed": args.seed,
                    "n_shot": args.n_shot,
                    "adaptation": args.adaptation,
                    "inner_lr": args.inner_lr,
                    "domain_strength": args.domain_strength,
                    "support_aug_strength": args.support_aug_strength,
                    "clean_outer_weight": args.clean_outer_weight,
                    "meta_shift": True,
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
