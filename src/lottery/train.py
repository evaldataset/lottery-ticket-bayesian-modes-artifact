from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader

from lottery.masks import Mask, apply_mask_, mask_gradients_


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float = 0.0,
    mask: Mapping[str, torch.Tensor] | None = None,
    lr_schedule: str = "constant",
    lr_schedule_epochs: int | None = None,
) -> None:
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )
    if lr_schedule == "constant":
        scheduler = None
    elif lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, lr_schedule_epochs or epochs),
        )
    else:
        raise ValueError(f"Unsupported lr_schedule: {lr_schedule}")
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            if mask is not None:
                mask_gradients_(model, mask)
            optimizer.step()
            if mask is not None:
                apply_mask_(model, mask)
        if scheduler is not None:
            scheduler.step()


@torch.no_grad()
def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.numel()
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


@torch.no_grad()
def predictions(model: nn.Module, data_loader: DataLoader, device: torch.device) -> torch.Tensor:
    model.to(device)
    model.eval()
    chunks = []
    for x, _ in data_loader:
        logits = model(x.to(device))
        chunks.append(logits.argmax(dim=1).cpu())
    return torch.cat(chunks)


@torch.no_grad()
def logits_matrix(model: nn.Module, data_loader: DataLoader, device: torch.device) -> torch.Tensor:
    model.to(device)
    model.eval()
    chunks = []
    for x, _ in data_loader:
        chunks.append(model(x.to(device)).cpu())
    return torch.cat(chunks)


def load_trainable_state(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    model.load_state_dict({key: value.detach().clone() for key, value in state.items()})


def state_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
