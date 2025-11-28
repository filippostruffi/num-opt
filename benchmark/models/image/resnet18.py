from typing import Any
from torch import nn
from torchvision import models
from benchmark.config.registry import register_model
from benchmark.models.base_model import infer_num_classes_from_dataset


@register_model("resnet18")
def build_model(task: Any, dataset: Any) -> nn.Module:
    # Build randomly initialized ResNet18 classifier
    num_classes = infer_num_classes_from_dataset(dataset, default=10)
    m = models.resnet18(weights=None)
    # Adapt first conv for input channels (e.g., MNIST 1-channel) and small images
    x, _ = dataset[0]
    in_ch = x.shape[0]
    if in_ch != 3:
        # Use CIFAR-style stem for small/tiny images and non-RGB inputs
        m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


