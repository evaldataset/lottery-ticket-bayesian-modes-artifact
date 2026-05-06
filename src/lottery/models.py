from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        depth: int = 3,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2")
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyCNN(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], num_classes: int, width: int = 32) -> None:
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError("TinyCNN expects image input_shape=(C,H,W)")
        channels = input_shape[0]
        self.features = nn.Sequential(
            nn.Conv2d(channels, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        pooled_h = input_shape[1] // 4
        pooled_w = input_shape[2] // 4
        self.classifier = nn.Linear(width * 2 * pooled_h * pooled_w, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(torch.flatten(x, start_dim=1))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNetCIFAR(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, ...],
        num_classes: int,
        blocks_per_stage: int = 3,
        width: int = 16,
    ) -> None:
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError("ResNetCIFAR expects image input_shape=(C,H,W)")
        self.in_planes = width
        self.conv1 = nn.Conv2d(input_shape[0], width, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.layer1 = self._make_layer(width, blocks_per_stage, stride=1)
        self.layer2 = self._make_layer(width * 2, blocks_per_stage, stride=2)
        self.layer3 = self._make_layer(width * 4, blocks_per_stage, stride=2)
        self.fc = nn.Linear(width * 4, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, block_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = torch.flatten(out, start_dim=1)
        return self.fc(out)


def weight_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, param in model.named_parameters() if param.ndim > 1]
