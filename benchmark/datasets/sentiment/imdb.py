from typing import Tuple, Optional
from datasets import load_dataset
from torch.utils.data import Dataset
from benchmark.datasets.base_dataset import maybe_limit_split
from benchmark.config.registry import register_dataset


class IMDBDataset(Dataset):
    def __init__(self, hf_split):
        self.data = hf_split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["text"], int(item["label"])


@register_dataset("imdb")
def build_imdb(max_samples: Optional[int], data_root: str) -> Tuple[object, object]:
    ds = load_dataset("imdb", cache_dir=f"{data_root}/imdb")
    train = IMDBDataset(ds["train"])
    val = IMDBDataset(ds["test"])
    return maybe_limit_split(train, val, max_samples)


