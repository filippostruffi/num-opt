from typing import Tuple, Optional
import os
import hashlib
import tarfile
from urllib.request import urlopen, Request
import ssl
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from benchmark.datasets.base_dataset import maybe_limit_split
from benchmark.config.registry import register_dataset


class VOC2012Segmentation(Dataset):
    """
    Pascal VOC2012 semantic segmentation.
    Uses torchvision.datasets.VOCSegmentation with auto-download support.
    Returns (image, mask) with:
      - image: normalized to ImageNet stats, resized to 256x256
      - mask: long tensor with class indices, resized with NEAREST
    Known properties:
      - num_classes = 21 (including background)
      - ignore_index = 255
    """
    def __init__(self, root: str, split: str = "train"):
        assert split in ("train", "val"), "split must be 'train' or 'val'"
        year = "2012"
        image_set = "train" if split == "train" else "val"
        # VOCSegmentation expects root pointing to the parent where VOCdevkit/ will be created
        tx = transforms.Compose(
            [
                transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        ty = transforms.Compose(
            [
                transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
                transforms.PILToTensor(),
            ]
        )
        try:
            self.ds = VOCSegmentation(
                root=root,
                year=year,
                image_set=image_set,
                download=True,
                transform=tx,
                target_transform=ty,
            )
        except Exception:
            # Fallback: manual mirror download if primary URL/md5 fails
            self._ensure_voc_present(root)
            self.ds = VOCSegmentation(
                root=root,
                year=year,
                image_set=image_set,
                download=False,
                transform=tx,
                target_transform=ty,
            )
        self.num_classes = 21
        self.ignore_index = 255

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image, mask = self.ds[idx]
        # mask is 1xHxW uint8; convert to long class indices
        mask = mask.squeeze(0).long()
        return image, mask

    @staticmethod
    def _ensure_voc_present(root: str) -> None:
        os.makedirs(root, exist_ok=True)
        expected_dir = os.path.join(root, "VOCdevkit", "VOC2012")
        if os.path.isdir(expected_dir):
            return
        filename = "VOCtrainval_11-May-2012.tar"
        dest_tar = os.path.join(root, filename)
        mirrors = [
            f"http://host.robots.ox.ac.uk/pascal/VOC/voc2012/{filename}",
            f"https://pjreddie.com/media/files/{filename}",
            f"https://archive.org/download/pascal-voc-2012/{filename}",
            f"https://thor.robots.ox.ac.uk/datasets/pascal/VOC/voc2012/{filename}",
        ]
        md5_expected = "6cd6e144f989b92b3379bac3b3de84fd"
        # If the tar already exists (e.g., downloaded via curl), try to extract it first
        if os.path.exists(dest_tar):
            try:
                with tarfile.open(dest_tar) as tf:
                    tf.extractall(path=root)
            except Exception:
                # If extraction fails, we'll try to re-download below
                pass
            if os.path.isdir(expected_dir):
                return
        # Download from first working mirror
        for url in mirrors:
            try:
                req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urlopen(req, timeout=300, context=ssl.create_default_context()) as resp:
                    data = resp.read()
                with open(dest_tar, "wb") as f:
                    f.write(data)
                # Verify md5
                h = hashlib.md5()
                with open(dest_tar, "rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
                if h.hexdigest() != md5_expected:
                    # Corrupt; try next mirror
                    try:
                        os.remove(dest_tar)
                    except Exception:
                        pass
                    continue
                # Extract
                with tarfile.open(dest_tar) as tf:
                    tf.extractall(path=root)
                break
            except Exception:
                # Try next mirror
                continue
        # Ensure extracted dir exists
        if not os.path.isdir(expected_dir):
            raise RuntimeError(
                f"Failed to prepare VOC2012 dataset under {root}. "
                f"Tried mirrors: {', '.join(mirrors)}. "
                "You can also download manually:\n"
                f"  mkdir -p {root} && cd {root}\n"
                f"  curl -L -o {filename} http://host.robots.ox.ac.uk/pascal/VOC/voc2012/{filename}\n"
                f"  tar -xf {filename}"
            )


@register_dataset("voc2012")
def build_voc2012(max_samples: Optional[int], data_root: str) -> Tuple[object, object]:
    # Store under <data_root>/voc2012 (VOCSegmentation will create VOCdevkit/VOC2012 under this)
    root = os.path.join(data_root, "voc2012")
    train = VOC2012Segmentation(root=root, split="train")
    val = VOC2012Segmentation(root=root, split="val")
    return maybe_limit_split(train, val, max_samples)


