"""Class-folder datasets and episodic sampling."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

from configs.experiment import IMAGENET_MEAN, IMAGENET_STD, NUM_WORKERS

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class ClassFolderDataset(Dataset):
    """Read images from ``root/class_name/image`` folders."""

    def __init__(self, root: str | Path, transform=None) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        self.transform = transform
        self.classes = sorted(
            path.name for path in self.root.iterdir() if path.is_dir()
        )
        if not self.classes:
            raise ValueError(f"No class folders found under {self.root}")
        self.class_to_index = {name: index for index, name in enumerate(self.classes)}
        self.images: list[Path] = []
        self.labels: list[int] = []
        for class_name in self.classes:
            class_dir = self.root / class_name
            paths = sorted(
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            for path in paths:
                self.images.append(path)
                self.labels.append(self.class_to_index[class_name])
        if not self.images:
            raise ValueError(f"No images found under {self.root}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = Image.open(self.images[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]


class EpisodicSampler(Sampler[list[int]]):
    def __init__(
        self,
        labels,
        n_way: int,
        n_shot: int,
        n_query: int,
        n_episodes: int,
        *,
        seed: int | None = None,
    ) -> None:
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        self.rng = random.Random(seed) if seed is not None else None
        self.class_indices = defaultdict(list)
        for index, label in enumerate(labels):
            self.class_indices[label].append(index)
        self.classes = list(self.class_indices)
        required = n_shot + n_query
        too_small = [
            label
            for label, indices in self.class_indices.items()
            if len(indices) < required
        ]
        if too_small:
            raise ValueError(
                f"Classes with fewer than {required} images: {too_small[:10]}"
            )
        if len(self.classes) < n_way:
            raise ValueError(
                f"Need at least {n_way} classes, found {len(self.classes)}"
            )

    def __len__(self) -> int:
        return self.n_episodes

    def __iter__(self):
        rng = self.rng or random
        for _ in range(self.n_episodes):
            chosen_classes = rng.sample(self.classes, self.n_way)
            episode = []
            for class_index in chosen_classes:
                episode.extend(
                    rng.sample(
                        self.class_indices[class_index],
                        self.n_shot + self.n_query,
                    )
                )
            yield episode


class EpisodicCollate:
    def __init__(
        self, n_way: int, n_shot: int, n_query: int, *, shuffle_queries: bool = False
    ) -> None:
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.shuffle_queries = bool(shuffle_queries)

    def __call__(self, batch):
        images = torch.stack([item[0] for item in batch])
        per_class = self.n_shot + self.n_query
        support_images, query_images = [], []
        support_labels, query_labels = [], []
        for class_index in range(self.n_way):
            start = class_index * per_class
            class_images = images[start : start + per_class]
            support_images.append(class_images[: self.n_shot])
            query_images.append(class_images[self.n_shot :])
            support_labels.extend([class_index] * self.n_shot)
            query_labels.extend([class_index] * self.n_query)
        support_tensor = torch.cat(support_images)
        query_tensor = torch.cat(query_images)
        support_label_tensor = torch.tensor(support_labels, dtype=torch.long)
        query_label_tensor = torch.tensor(query_labels, dtype=torch.long)

        # The canonical episodic layout is class-major.  That ordering is harmless
        # for single-query methods, but a multi-query method would otherwise obtain
        # groups containing only one class (an accidental label-order leak).  When
        # requested, shuffle query samples and labels together using the DataLoader
        # worker RNG.  Group membership is therefore independent of class position.
        if self.shuffle_queries and query_tensor.shape[0] > 1:
            permutation = torch.randperm(query_tensor.shape[0])
            query_tensor = query_tensor[permutation]
            query_label_tensor = query_label_tensor[permutation]

        return (
            support_tensor,
            support_label_tensor,
            query_tensor,
            query_label_tensor,
        )


def evaluation_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def training_transform():
    """Training augmentation used by the original WIPT training scripts."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def make_episode_loader(
    data_dir: str | Path,
    *,
    n_way: int,
    n_shot: int,
    n_query: int,
    n_episodes: int,
    transform=None,
    num_workers: int = NUM_WORKERS,
    seed: int | None = None,
    shuffle_queries: bool = False,
) -> DataLoader:
    dataset = ClassFolderDataset(
        data_dir, transform=transform or evaluation_transform()
    )
    sampler = EpisodicSampler(
        dataset.labels,
        n_way,
        n_shot,
        n_query,
        n_episodes,
        seed=seed,
    )
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=EpisodicCollate(
            n_way, n_shot, n_query, shuffle_queries=shuffle_queries
        ),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
