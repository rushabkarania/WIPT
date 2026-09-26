"""Evaluate the A/B/C corruption protocol used in the paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from configs.experiment import (
    NUM_WORKERS,
    CORRUPTION_EPISODES_PER_SEED,
    DEFAULT_DATA_PATHS,
    EVAL_SEEDS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    N_QUERY,
    N_SHOT,
    N_WAY,
)
from eval.runtime import get_device, load_paper_models
from utils.data import ClassFolderDataset, EpisodicCollate, EpisodicSampler
from utils.metrics import accuracy, mean_ci95, set_seed

DOMAINS = ("CUB", "EuroSAT", "ISIC")
SEVERITIES = (1, 2, 3, 4, 5)
CATEGORIES = {
    "A_structured": ("brightness", "contrast"),
    "B_stochastic": ("gaussian_noise",),
    "C_combined": ("brightness", "contrast", "gaussian_noise"),
}
BRIGHTNESS = (0.10, 0.20, 0.30, 0.40, 0.50)
CONTRAST = (0.75, 0.60, 0.50, 0.40, 0.30)
GAUSSIAN_SIGMA = (0.08, 0.12, 0.18, 0.26, 0.38)
NORMALIZE = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
TO_TENSOR = transforms.ToTensor()
RESIZE = transforms.Resize(256)
CROP = transforms.CenterCrop(224)


def apply_brightness(image: np.ndarray, severity: int) -> np.ndarray:
    x = image.astype(np.float32) / 255.0
    return (np.clip(x + BRIGHTNESS[severity - 1], 0, 1) * 255).astype(np.uint8)


def apply_contrast(image: np.ndarray, severity: int) -> np.ndarray:
    x = image.astype(np.float32) / 255.0
    mean = x.mean(axis=(0, 1), keepdims=True)
    x = np.clip((x - mean) * CONTRAST[severity - 1] + mean, 0, 1)
    return (x * 255).astype(np.uint8)


def apply_gaussian_noise(image: np.ndarray, severity: int) -> np.ndarray:
    x = image.astype(np.float32) / 255.0
    noise = np.random.normal(0.0, GAUSSIAN_SIGMA[severity - 1], size=x.shape)
    return (np.clip(x + noise, 0, 1) * 255).astype(np.uint8)


CORRUPTIONS = {
    "brightness": apply_brightness,
    "contrast": apply_contrast,
    "gaussian_noise": apply_gaussian_noise,
}


class CorruptedDataset(ClassFolderDataset):
    def __init__(self, root, corruption_names=(), severity=1):
        super().__init__(root, transform=None)
        self.corruption_names = tuple(corruption_names)
        self.severity = severity

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        image = CROP(RESIZE(image))
        array = np.asarray(image, dtype=np.uint8)
        for name in self.corruption_names:
            array = CORRUPTIONS[name](array, self.severity)
        return NORMALIZE(TO_TENSOR(Image.fromarray(array))), self.labels[index]


def make_loader(data_path, corruption_names, severity, seed):
    set_seed(seed)
    dataset = CorruptedDataset(data_path, corruption_names, severity)
    sampler = EpisodicSampler(
        dataset.labels,
        N_WAY,
        N_SHOT,
        N_QUERY,
        CORRUPTION_EPISODES_PER_SEED,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=EpisodicCollate(N_WAY, N_SHOT, N_QUERY),
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate_condition(model, data_path, corruption_names, severity, device):
    values = []
    for seed in EVAL_SEEDS:
        # Resetting each model/seed keeps both episode sampling and Gaussian
        # noise realizations matched across methods.
        loader = make_loader(data_path, corruption_names, severity, seed)
        with torch.no_grad():
            for support, _, query, labels in loader:
                support, query, labels = (
                    support.to(device),
                    query.to(device),
                    labels.to(device),
                )
                values.append(accuracy(model(support, query, N_WAY, N_SHOT), labels))
    return np.asarray(values, dtype=float) * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cub", default=DEFAULT_DATA_PATHS["CUB"])
    parser.add_argument("--eurosat", default=DEFAULT_DATA_PATHS["EuroSAT"])
    parser.add_argument("--isic", default=DEFAULT_DATA_PATHS["ISIC"])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/raw/corruption/abc_corruption.csv"),
    )
    args = parser.parse_args()

    device = get_device()
    all_models = load_paper_models(device, include_variants=False)
    models = {
        name: model
        for name, (model, _) in all_models.items()
        if name in {"ProtoNet+ViT", "SupportTransformer+ViT", "WIPT-2"}
    }
    missing = {"ProtoNet+ViT", "SupportTransformer+ViT", "WIPT-2"} - set(models)
    if missing:
        raise FileNotFoundError(
            f"Required models/checkpoints are unavailable: {sorted(missing)}"
        )

    data_paths = {"CUB": args.cub, "EuroSAT": args.eurosat, "ISIC": args.isic}
    rows = []
    for domain in DOMAINS:
        data_path = data_paths[domain]
        for model_name, model in models.items():
            values = evaluate_condition(model, data_path, (), 1, device)
            mean, ci = mean_ci95(values)
            rows.append(
                {
                    "dataset": domain,
                    "category": "clean",
                    "severity": 0,
                    "model": model_name,
                    "accuracy": mean,
                    "ci95": ci,
                    "n_episodes": len(values),
                }
            )
        for category, corruption_names in CATEGORIES.items():
            for severity in SEVERITIES:
                for model_name, model in models.items():
                    values = evaluate_condition(
                        model, data_path, corruption_names, severity, device
                    )
                    mean, ci = mean_ci95(values)
                    rows.append(
                        {
                            "dataset": domain,
                            "category": category,
                            "severity": severity,
                            "model": model_name,
                            "accuracy": mean,
                            "ci95": ci,
                            "n_episodes": len(values),
                        }
                    )
                print(f"{domain}: {category} severity {severity} complete")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"saved {args.out}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
