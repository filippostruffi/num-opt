from typing import Tuple, Optional
from torchvision import datasets, transforms
from benchmark.datasets.base_dataset import maybe_limit_split
from benchmark.config.registry import register_dataset


@register_dataset("cifar10")
def build_cifar10(max_samples: Optional[int], data_root: str) -> Tuple[object, object]:
    train_tfm = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ]
    )
    test_tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ]
    )
    root = f"{data_root}/cifar10"
    train = datasets.CIFAR10(root=root, train=True, download=True, transform=train_tfm)
    val = datasets.CIFAR10(root=root, train=False, download=True, transform=test_tfm)
    return maybe_limit_split(train, val, max_samples)


