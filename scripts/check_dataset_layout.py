"""Check that the datasets have the class-folder layout expected by the loaders."""

from __future__ import annotations

import argparse
from pathlib import Path

from configs.experiment import DEFAULT_DATA_PATHS, N_SHOT, N_QUERY, N_WAY

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def inspect_dataset(name: str, path: Path) -> tuple[int, int, int]:
    if not path.exists():
        raise FileNotFoundError(f"{name}: {path} does not exist")
    classes = sorted(p for p in path.iterdir() if p.is_dir())
    if len(classes) < N_WAY:
        raise ValueError(
            f"{name}: needs at least {N_WAY} class folders, found {len(classes)}"
        )

    image_counts = [
        sum(
            1
            for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        for class_dir in classes
    ]
    required = N_SHOT + N_QUERY
    too_small = sum(count < required for count in image_counts)
    if too_small:
        raise ValueError(
            f"{name}: {too_small} classes contain fewer than {required} images"
        )
    return len(classes), sum(image_counts), min(image_counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-optional", action="store_true")
    args = parser.parse_args()
    for name, raw_path in DEFAULT_DATA_PATHS.items():
        if name in {"CropDisease", "ChestX"} and not args.include_optional:
            continue
        classes, images, minimum = inspect_dataset(name, Path(raw_path))
        print(
            f"{name}: {classes} classes, {images} images, minimum {minimum} images/class"
        )


if __name__ == "__main__":
    main()
