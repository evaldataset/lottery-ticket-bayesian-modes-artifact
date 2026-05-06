from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset, TensorDataset


@dataclass(frozen=True)
class DatasetBundle:
    train_loader: DataLoader
    test_loader: DataLoader
    input_dim: int
    input_shape: tuple[int, ...]
    num_classes: int
    train_size: int
    test_size: int


def load_digits_bundle(
    batch_size: int,
    test_batch_size: int,
    seed: int,
    test_size: float = 0.2,
) -> DatasetBundle:
    digits = load_digits()
    x = digits.data.astype(np.float32)
    y = digits.target.astype(np.int64)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False)
    return DatasetBundle(
        train_loader=train_loader,
        test_loader=test_loader,
        input_dim=x.shape[1],
        input_shape=(x.shape[1],),
        num_classes=int(y.max()) + 1,
        train_size=len(train_ds),
        test_size=len(test_ds),
    )


def load_torchvision_bundle(
    dataset: str,
    batch_size: int,
    test_batch_size: int,
    seed: int,
    root: str | Path = "data",
    flatten: bool = True,
    train_subset: int | None = None,
    test_subset: int | None = None,
    augment: bool = False,
) -> DatasetBundle:
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required for MNIST/Fashion-MNIST/CIFAR-10. "
            "Use .venv/bin/python after creating the project venv."
        ) from exc

    dataset_key = dataset.lower().replace("_", "-")
    if dataset_key == "mnist":
        dataset_cls = datasets.MNIST
        mean = (0.1307,)
        std = (0.3081,)
        num_classes = 10
    elif dataset_key in {"fashion-mnist", "fashion"}:
        dataset_cls = datasets.FashionMNIST
        mean = (0.2860,)
        std = (0.3530,)
        num_classes = 10
    elif dataset_key in {"cifar10", "cifar-10"}:
        dataset_cls = datasets.CIFAR10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        num_classes = 10
    else:
        raise ValueError(f"Unsupported torchvision dataset: {dataset}")

    train_transform_steps = []
    if augment:
        if dataset_key not in {"cifar10", "cifar-10"}:
            raise ValueError("augment=True is currently supported only for CIFAR-10")
        if flatten:
            raise ValueError("CIFAR augmentation requires image tensors; set flatten=False")
        train_transform_steps.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    transform_steps = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    if flatten:
        transform_steps.append(transforms.Lambda(lambda tensor: tensor.flatten()))
    train_transform = transforms.Compose([*train_transform_steps, *transform_steps])
    test_transform = transforms.Compose(transform_steps)
    train_ds = dataset_cls(root=root, train=True, download=True, transform=train_transform)
    test_ds = dataset_cls(root=root, train=False, download=True, transform=test_transform)
    if train_subset is not None:
        train_count = min(train_subset, len(train_ds))
        train_ds = Subset(train_ds, list(range(train_count)))
    if test_subset is not None:
        test_count = min(test_subset, len(test_ds))
        test_ds = Subset(test_ds, list(range(test_count)))
    sample, _ = train_ds[0]

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return DatasetBundle(
        train_loader=train_loader,
        test_loader=test_loader,
        input_dim=int(sample.numel()),
        input_shape=tuple(sample.shape),
        num_classes=num_classes,
        train_size=len(train_ds),
        test_size=len(test_ds),
    )


def load_fake_cifar10_bundle(
    batch_size: int,
    test_batch_size: int,
    seed: int,
    train_size: int = 2048,
    test_size: int = 512,
) -> DatasetBundle:
    generator = torch.Generator()
    generator.manual_seed(seed)
    x_train = torch.randn(train_size, 3, 32, 32, generator=generator)
    y_train = torch.randint(0, 10, (train_size,), generator=generator)
    x_test = torch.randn(test_size, 3, 32, 32, generator=generator)
    y_test = torch.randint(0, 10, (test_size,), generator=generator)
    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return DatasetBundle(
        train_loader=train_loader,
        test_loader=test_loader,
        input_dim=3 * 32 * 32,
        input_shape=(3, 32, 32),
        num_classes=10,
        train_size=train_size,
        test_size=test_size,
    )
