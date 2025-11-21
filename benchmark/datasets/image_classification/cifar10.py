from typing import Tuple, Optional
from torchvision import datasets, transforms
from benchmark.datasets.base_dataset import maybe_limit_split
from benchmark.config.registry import register_dataset


@register_dataset("cifar10")
def build_cifar10(max_samples: Optional[int], data_root: str) -> Tuple[object, object]:
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    root = f"{data_root}/cifar10"
    train = datasets.CIFAR10(root=root, train=True, download=True, transform=tfm)
    val = datasets.CIFAR10(root=root, train=False, download=True, transform=tfm)
    return maybe_limit_split(train, val, max_samples)


