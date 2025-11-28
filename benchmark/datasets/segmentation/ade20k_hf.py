from typing import Tuple, Optional
import os
from datasets import load_dataset  # type: ignore
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch
from benchmark.datasets.base_dataset import maybe_limit_split
from benchmark.config.registry import register_dataset


class ADE20KDataset(Dataset):
    def __init__(self, split: str = "train", cache_dir: Optional[str] = None):
        # Load ADE20K / SceneParse150 from the parquet-only clone on HF
        last_err: Optional[Exception] = None
        self.ds = None

        try:
            # merve/scene_parse_150:
            #   subset: default
            #   splits: train / validation / test
            #   columns: image, annotation, scene_category
            self.ds = load_dataset(
                "merve/scene_parse_150",
                split=split,
                cache_dir=cache_dir,
            )
        except Exception as ex:
            last_err = ex

        if self.ds is None:
            raise RuntimeError(
                "Unable to load ADE20K (merve/scene_parse_150) from Hugging Face.\n"
                "Check your internet connection and `datasets` installation.\n"
                f"Last error: {last_err}"
            )

        self.img_tfm = transforms.Compose(
            [
                transforms.Lambda(lambda im: im.convert("RGB")),
                transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.mask_tfm = transforms.Compose(
            [
                transforms.Lambda(lambda m: m.convert("L")),
                transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
            ]
        )
        # Known dataset properties (150 foreground classes + background = 151)
        self.num_classes = 151
        self.ignore_index = 255


    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex["image"]          
        mask = ex["annotation"]    

        img = self.img_tfm(img)
        mask = self.mask_tfm(mask)
        mask = transforms.PILToTensor()(mask).squeeze(0).long()

        return img, mask


@register_dataset("ade20k")
def build_ade20k(max_samples: Optional[int], data_root: str) -> Tuple[object, object]:
    cache_dir = os.path.join(data_root, "ade20k")
    train = ADE20KDataset(split="train", cache_dir=cache_dir)
    val = ADE20KDataset(split="validation", cache_dir=cache_dir)
    return maybe_limit_split(train, val, max_samples)
