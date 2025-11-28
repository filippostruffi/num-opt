from typing import Tuple, Optional
import os
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
                    # Normalize to ImageNet stats to stabilize optimization across models
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
            target_transform=transforms.Compose(
                [
                    transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
                    transforms.PILToTensor(),
                ]
            ),
        )
        # Filter out any entries with missing image/trimap files to avoid runtime crashes
        try:
            images = getattr(self.ds, "_images", [])
            segs = getattr(self.ds, "_segs", [])
            if images and segs and len(images) == len(segs):
                self._valid_indices = [
                    i for i, (ip, sp) in enumerate(zip(images, segs)) if os.path.exists(ip) and os.path.exists(sp)
                ]
            else:
                self._valid_indices = None
        except Exception:
            self._valid_indices = None

    def __len__(self):
        if self._valid_indices is not None:
            return len(self._valid_indices)
        return len(self.ds)

    def __getitem__(self, idx):
        if self._valid_indices is not None:
            idx = self._valid_indices[idx]
        image, mask = self.ds[idx]
        # mask is 1xHxW with values in {1,2,3}, where 1=pet, 2=outline, 3=background (palette-based).
        # Convert to 0-based class indices in {0,1,2} and keep 255 as ignore if present.
        mask = mask.squeeze(0).long()
        mask = torch.where(mask == 255, mask, mask - 1)  # 1->0, 2->1, 3->2
        # Common practice for this dataset: ignore the thin "outline" class and
        # collapse to foreground (pet) vs background. Map:
        #   outline (1) -> 255 (ignore), background (2) -> 1, pet (0) -> 0
        mask = torch.where(mask == 1, torch.tensor(255, dtype=mask.dtype, device=mask.device), mask)
        mask = torch.where(mask == 2, torch.tensor(1, dtype=mask.dtype, device=mask.device), mask)
        return image, mask


@register_dataset("oxford_iiit_pet")
def build_oxford_pet(max_samples: Optional[int], data_root: str) -> Tuple[object, object]:
    root = f"{data_root}/oxford_iiit_pet"
    train = OxfordPetSegmentation(root=root, split="trainval")
    val = OxfordPetSegmentation(root=root, split="test")
    return maybe_limit_split(train, val, max_samples)


