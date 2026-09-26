"""Preflight check for the training and evaluation inputs."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch

from configs.experiment import DEFAULT_DATA_PATHS, TRAIN_SEEDS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint-root", type=Path, default=Path("checkpoints/seed_study")
    )
    p.add_argument("--require-extra-targets", action="store_true")
    a = p.parse_args()
    rows = []
    rows.append(("python torch", True, torch.__version__))
    rows.append(
        (
            "timm",
            importlib.util.find_spec("timm") is not None,
            "required for model runs",
        )
    )
    rows.append(
        (
            "CUDA",
            torch.cuda.is_available(),
            (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "CPU-only; training will be very slow"
            ),
        )
    )
    for key in ["miniImageNet_train", "miniImageNet_val", "CUB", "EuroSAT", "ISIC"]:
        path = Path(DEFAULT_DATA_PATHS[key])
        rows.append((f"dataset {key}", path.exists(), str(path)))
    if a.require_extra_targets:
        for key in ["CropDisease", "ChestX"]:
            path = Path(DEFAULT_DATA_PATHS[key])
            rows.append((f"dataset {key}", path.exists(), str(path)))
    for shot in [1, 5]:
        for seed in TRAIN_SEEDS:
            d = a.checkpoint_root / f"{shot}shot" / f"seed_{seed}"
            for name in ["wipt_l2.pth", "support_transformer.pth"]:
                path = d / name
                rows.append(
                    (
                        f"checkpoint {shot}shot seed{seed} {name}",
                        path.exists(),
                        str(path),
                    )
                )
    width = max(len(r[0]) for r in rows)
    for name, ok, detail in rows:
        print(f"{'OK' if ok else 'MISSING':7} {name:<{width}}  {detail}")
    required = [
        r
        for r in rows
        if not r[1]
        and (
            r[0] == "timm"
            or r[0].startswith("dataset miniImageNet")
            or r[0] in {"dataset CUB", "dataset EuroSAT", "dataset ISIC"}
            or r[0].startswith("checkpoint ")
            or (
                a.require_extra_targets
                and r[0] in {"dataset CropDisease", "dataset ChestX"}
            )
        )
    ]
    if required:
        print(
            f"\nPreflight: {len(required)} required item(s) missing. Model evaluations cannot run yet."
        )
        raise SystemExit(2)
    print("\nPreflight passed: model evaluations are ready to run.")


if __name__ == "__main__":
    main()
