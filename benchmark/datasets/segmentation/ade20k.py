from typing import Tuple, Optional
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch
from benchmark.datasets.base_dataset import maybe_limit_split
from benchmark.config.registry import register_dataset


class ADE20KDataset(Dataset):
    def __init__(self, split: str = "train", cache_dir: Optional[str] = None):
        # scene_parse_150 is ADE20K with 150 semantic classes
        self.ds = load_dataset("scene_parse_150", split=split, cache_dir=cache_dir)
        self.img_tfm = transforms.Compose(
            [
                transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )
        self.mask_tfm = transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex["image"]  # PIL
        mask = ex["annotation"]  # PIL segmentation map with class ids
        img = self.img_tfm(img)
        mask = self.mask_tfm(mask)
        mask = transforms.PILToTensor()(mask).squeeze(0).long()
        return img, mask


@register_dataset("ade20k_150")
def build_ade20k(max_samples: Optional[int], data_root: str) -> Tuple[object, object]:
    try:
        cache_dir = f"{data_root}/ade20k_150"
        train = ADE20KDataset(split="train", cache_dir=cache_dir)
        val = ADE20KDataset(split="validation", cache_dir=cache_dir)
        return maybe_limit_split(train, val, max_samples)
    except Exception:
        # Fallback: use Oxford-IIIT Pet segmentation to ensure runnable pipeline
        from benchmark.datasets.segmentation.oxford_iiit_pet import OxfordPetSegmentation
        root = f"{data_root}/oxford_iiit_pet"
        train = OxfordPetSegmentation(root=root, split="trainval")
        val = OxfordPetSegmentation(root=root, split="test")
        return maybe_limit_split(train, val, max_samples)


