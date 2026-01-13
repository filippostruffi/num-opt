from typing import Any
import torch
from torch import nn
from benchmark.config.registry import register_model
from benchmark.models.base_model import infer_num_classes_from_dataset


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7 if in_channels == 1 else 64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


@register_model("cnn_scratch")
def build_model(task: Any, dataset: Any) -> nn.Module:
    # infer channels and classes from a sample
    x, y = dataset[0]
    in_channels = x.shape[0]
    num_classes = infer_num_classes_from_dataset(dataset, default=10)
    return SmallCNN(in_channels=in_channels, num_classes=num_classes)


