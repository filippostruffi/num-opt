from typing import Tuple, Optional
from datasets import load_dataset
from torch.utils.data import Dataset
from benchmark.datasets.base_dataset import maybe_limit_split
from benchmark.config.registry import register_dataset


class SST2Dataset(Dataset):
    def __init__(self, hf_split, text_key="sentence", label_key="label"):
        self.data = hf_split
        self.text_key = text_key
        self.label_key = label_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item[self.text_key], int(item[self.label_key])


@register_dataset("sst2")
def build_sst2(max_samples: Optional[int], data_root: str) -> Tuple[object, object]:
    ds = load_dataset("glue", "sst2", cache_dir=f"{data_root}/sst2")
    train = SST2Dataset(ds["train"])
    val = SST2Dataset(ds["validation"])
    return maybe_limit_split(train, val, max_samples)


