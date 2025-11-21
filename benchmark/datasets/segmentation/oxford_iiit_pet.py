from typing import Tuple, Optional
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
import torch
from benchmark.datasets.base_dataset import maybe_limit_split
from benchmark.config.registry import register_dataset


class OxfordPetSegmentation(Dataset):
    def __init__(self, root: str, split: str = "trainval"):
        # target_types="segmentation" returns a per-pixel mask
        self.ds = OxfordIIITPet(
            root=root,
            split=split,
            target_types="segmentation",
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                ]
            ),
            target_transform=transforms.Compose(
                [
                    transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
                    transforms.PILToTensor(),
                ]
            ),
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image, mask = self.ds[idx]
        # mask is 1xHxW with values in {1,2,3}, where 1=pet, 2=outline, 3=background (palette-based).
        # Convert to 0-based class indices in {0,1,2} and keep 255 as ignore if present.
        mask = mask.squeeze(0).long()
        mask = torch.where(mask == 255, mask, mask - 1)  # 1->0, 2->1, 3->2
        return image, mask


@register_dataset("oxford_iiit_pet")
def build_oxford_pet(max_samples: Optional[int], data_root: str) -> Tuple[object, object]:
    root = f"{data_root}/oxford_iiit_pet"
    train = OxfordPetSegmentation(root=root, split="trainval")
    val = OxfordPetSegmentation(root=root, split="test")
    return maybe_limit_split(train, val, max_samples)


