"""Create the 64/16/20 miniImageNet class split used by the experiments."""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path


def collect_images(source: Path) -> dict[str, list[Path]]:
    classes: dict[str, list[Path]] = defaultdict(list)
    for image in sorted(source.glob("*.jpg")):
        classes[image.name[:9]].append(image)
    return dict(classes)


def prepare(source: Path, output: Path, *, overwrite: bool = False) -> None:
    if (
        output.resolve() == source.resolve()
        or output.resolve() in source.resolve().parents
    ):
        raise ValueError(
            "Output must not be the source directory or one of its parents."
        )
    class_images = collect_images(source)
    class_names = sorted(class_images)
    if len(class_names) != 100:
        raise ValueError(
            f"Expected 100 miniImageNet classes, found {len(class_names)} in {source}."
        )

    splits = {
        "train": class_names[:64],
        "val": class_names[64:80],
        "test": class_names[80:],
    }

    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output} already exists. Pass --overwrite to replace it."
            )
        shutil.rmtree(output)

    for split, classes in splits.items():
        for class_name in classes:
            destination = output / split / class_name
            destination.mkdir(parents=True, exist_ok=True)
            for image in class_images[class_name]:
                shutil.copy2(image, destination / image.name)

    total = sum(len(images) for images in class_images.values())
    print(
        f"Prepared miniImageNet at {output}: "
        f"64 train / 16 val / 20 test classes, {total} images."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/miniimagenet"))
    parser.add_argument("--output", type=Path, default=Path("data/miniimagenet_split"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare(args.source, args.output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
