"""Shared episodic training utilities."""

from __future__ import annotations

from pathlib import Path
import csv
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from configs.experiment import (
    EPOCHS,
    ETA_MIN,
    LEARNING_RATE,
    N_QUERY,
    N_SHOT,
    N_WAY,
    NUM_WORKERS,
    TRAIN_EPISODES,
    VAL_EPISODES,
)
from utils.data import make_episode_loader, evaluation_transform, training_transform
from utils.metrics import accuracy, mean_ci95


def make_train_val_loaders(
    train_dir,
    val_dir,
    num_workers: int = NUM_WORKERS,
    *,
    n_way: int = N_WAY,
    n_shot: int = N_SHOT,
    n_query: int = N_QUERY,
    seed: int | None = None,
    shuffle_queries: bool = False,
):
    train_loader = make_episode_loader(
        train_dir,
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
        n_episodes=TRAIN_EPISODES,
        transform=training_transform(),
        num_workers=num_workers,
        seed=seed,
        shuffle_queries=shuffle_queries,
    )
    val_seed = None if seed is None else seed + 10_000
    val_loader = make_episode_loader(
        val_dir,
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
        n_episodes=VAL_EPISODES,
        transform=evaluation_transform(),
        num_workers=num_workers,
        seed=val_seed,
        shuffle_queries=shuffle_queries,
    )
    return train_loader, val_loader


def train_epoch(
    model, loader, optimizer, device, loss_fn=None, *, n_way=N_WAY, n_shot=N_SHOT
):
    model.train()
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    losses, accuracies = [], []
    for support, _, query, labels in tqdm(loader, desc="train", leave=False):
        support, query, labels = support.to(device), query.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        scores = model(support, query, n_way, n_shot)
        loss = loss_fn(scores, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        accuracies.append(accuracy(scores, labels))
    return float(np.mean(losses)), float(np.mean(accuracies))


def evaluate(model, loader, device, *, n_way=N_WAY, n_shot=N_SHOT):
    model.eval()
    accuracies = []
    with torch.no_grad():
        for support, _, query, labels in loader:
            support, query, labels = (
                support.to(device),
                query.to(device),
                labels.to(device),
            )
            accuracies.append(accuracy(model(support, query, n_way, n_shot), labels))
    return mean_ci95(accuracies)


def fit(
    model,
    train_loader,
    val_loader,
    checkpoint_path,
    *,
    device,
    checkpoint_metadata=None,
    loss_fn=None,
    n_way: int = N_WAY,
    n_shot: int = N_SHOT,
    history_path: str | Path | None = None,
):
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("This model has no trainable parameters.")
    optimizer = Adam(trainable, lr=LEARNING_RATE, weight_decay=0.0)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=ETA_MIN)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = -float("inf")
    best_epoch = None
    best_elapsed = None
    history = []
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_fn=loss_fn,
            n_way=n_way,
            n_shot=n_shot,
        )
        val_acc, val_ci = evaluate(
            model, val_loader, device, n_way=n_way, n_shot=n_shot
        )
        scheduler.step()
        elapsed = time.time() - started
        eta_hours = (EPOCHS - epoch) * (elapsed / epoch) / 3600
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "validation_accuracy": val_acc,
                "validation_ci95": val_ci,
                "elapsed_seconds": elapsed,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"epoch {epoch:02d}/{EPOCHS} | loss {train_loss:.4f} | "
            f"train {100*train_acc:.2f}% | val {100*val_acc:.2f}% ± {100*val_ci:.2f}% | "
            f"eta {eta_hours:.1f} h"
        )
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_elapsed = elapsed
            payload = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_acc": best_val,
                "val_ci": val_ci,
                "elapsed_seconds_to_best": elapsed,
            }
            if checkpoint_metadata:
                payload.update(checkpoint_metadata)
            torch.save(payload, checkpoint_path)
            print(f"saved {checkpoint_path}")

    if history_path is not None:
        history_path = Path(history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    return best_val
