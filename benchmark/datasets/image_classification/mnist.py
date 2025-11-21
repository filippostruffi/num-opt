from typing import Tuple, Optional
import os
from torchvision import datasets, transforms
from benchmark.datasets.base_dataset import maybe_limit_split
from benchmark.config.registry import register_dataset


@register_dataset("mnist")
def build_mnist(max_samples: Optional[int], data_root: str) -> Tuple[object, object]:
    root = os.path.join(data_root if data_root is not None else "data", "mnist")
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    val = datasets.MNIST(root=root, train=False, download=True, transform=tfm)
    return maybe_limit_split(train, val, max_samples)


