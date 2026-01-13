from typing import Tuple, Optional, List, Dict
from datasets import load_dataset
from torch.utils.data import Dataset
from benchmark.datasets.base_dataset import maybe_limit_split
from benchmark.config.registry import register_dataset
import torch


def build_vocab_from_text(lines: List[str], max_size: int = 50000) -> Dict[str, int]:
    from collections import Counter
    counter = Counter()
    for s in lines:
        counter.update(s.strip().split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, _ in counter.most_common(max_size - len(vocab)):
        if w not in vocab:
            vocab[w] = len(vocab)
    return vocab


def encode_line(s: str, vocab: Dict[str, int]) -> List[int]:
    return [vocab.get(tok, 1) for tok in s.strip().split()]


class WikitextTokens(Dataset):
    def __init__(self, lines: List[str], vocab: Dict[str, int], block_size: int = 64):
        self.block_size = block_size
        self.vocab = vocab
        ids = []
        for line in lines:
            ids.extend(encode_line(line, vocab))
        # create sequences of block_size
        self.ids = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, (self.ids.size(0) - 1) // self.block_size)

    def __getitem__(self, idx):
        start = idx * self.block_size
        x = self.ids[start : start + self.block_size]
        y = self.ids[start + 1 : start + 1 + self.block_size]
        if y.size(0) < x.size(0):
            # pad last
            y = torch.cat([y, torch.zeros(x.size(0) - y.size(0), dtype=torch.long)], dim=0)
        return x, y


@register_dataset("wikitext2")
def build_wikitext2(max_samples: Optional[int], data_root: str) -> Tuple[object, object]:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=f"{data_root}/wikitext2")
    train_lines = ds["train"]["text"]
    val_lines = ds["validation"]["text"]
    vocab = build_vocab_from_text(train_lines)
    train = WikitextTokens(train_lines, vocab)
    val = WikitextTokens(val_lines, vocab)
    # attach vocab for model
    train.vocab = vocab
    val.vocab = vocab
    return maybe_limit_split(train, val, max_samples)


