import torch.nn as nn
from torchvision.models import resnet18
from typing import cast
from torch import compile as torch_compile


class SimpleCNN(nn.Module):
    """Простая сверточная нейронная сеть для CIFAR-10."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(8 * 8 * 64, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def get_model(model_name: str, device: str) -> nn.Module:
    """Возвращает скомпилированную модель."""
    if model_name == "simplecnn":
        model = SimpleCNN()
    elif model_name == "resnet18":
        model = resnet18(weights=None)
        # Настройка ResNet18 для CIFAR-10 (32x32 изображения)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, 10)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model = model.to(device)
    # Используем torch.compile для ускорения (требует PyTorch >= 2.0)
    compiled_model = torch_compile(model)
    return cast(nn.Module, compiled_model)
