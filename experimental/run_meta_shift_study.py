"""Train MAML-inspired WIPT and support-only heads over independent seeds."""

from __future__ import annotations
import argparse
from pathlib import Path
import subprocess, sys
import pandas as pd
from configs.experiment import DEFAULT_DATA_PATHS, TRAIN_SEEDS, NUM_WORKERS
from experiments.seed_study_common import parse_seeds, shot_tag
from utils.checkpoints import load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shot", type=int, default=5, choices=[1, 5])
    p.add_argument("--seeds", type=int, nargs="*", default=list(TRAIN_SEEDS))
    p.add_argument("--workers", type=int, default=NUM_WORKERS)
    p.add_argument("--train-dir", default=DEFAULT_DATA_PATHS["miniImageNet_train"])
    p.add_argument("--val-dir", default=DEFAULT_DATA_PATHS["miniImageNet_val"])
    p.add_argument(
        "--checkpoint-root", type=Path, default=Path("checkpoints/meta_shift")
    )
    p.add_argument("--out-root", type=Path, default=Path("results/raw/meta_shift"))
    p.add_argument("--inner-lr", type=float, default=1e-3)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    seeds = parse_seeds(a.seeds)
    rows = []
    for adaptation in ["joint", "support"]:
        for seed in seeds:
            d = a.checkpoint_root / shot_tag(a.shot) / adaptation / f"seed_{seed}"
            d.mkdir(parents=True, exist_ok=True)
            ck = d / "model.pth"
            hist = d / "history.csv"
            if not ck.exists() or a.overwrite:
                cmd = [
                    sys.executable,
                    "-m",
                    "experimental.train_meta_shift",
                    "--adaptation",
                    adaptation,
                    "--seed",
                    str(seed),
                    "--n-shot",
                    str(a.shot),
                    "--workers",
                    str(a.workers),
                    "--train-dir",
                    str(a.train_dir),
                    "--val-dir",
                    str(a.val_dir),
                    "--inner-lr",
                    str(a.inner_lr),
                    "--output",
                    str(ck),
                    "--history",
                    str(hist),
                ]
                print(" ".join(cmd))
                subprocess.run(cmd, check=True)
            c = load_checkpoint(ck, "cpu")
            rows.append(
                {
                    "adaptation": adaptation,
                    "n_shot": a.shot,
                    "training_seed": seed,
                    "best_epoch": int(c["epoch"]),
                    "best_validation_accuracy": 100 * float(c["best_val_acc"]),
                    "elapsed_seconds_to_best": c.get("elapsed_seconds_to_best"),
                    "checkpoint": str(ck),
                }
            )
    out = a.out_root / shot_tag(a.shot)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "training_checkpoints.csv", index=False)


if __name__ == "__main__":
    main()
