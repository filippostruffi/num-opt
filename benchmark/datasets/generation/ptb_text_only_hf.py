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


class PTBTokens(Dataset):
    def __init__(self, lines: List[str], vocab: Dict[str, int], block_size: int = 64):
        self.block_size = block_size
        self.vocab = vocab
        ids: List[int] = []
        for line in lines:
            ids.extend(encode_line(line, vocab))
        self.ids = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, (self.ids.size(0) - 1) // self.block_size)

    def __getitem__(self, idx):
        start = idx * self.block_size
        x = self.ids[start : start + self.block_size]
        y = self.ids[start + 1 : start + 1 + self.block_size]
        if y.size(0) < x.size(0):
            y = torch.cat([y, torch.zeros(x.size(0) - y.size(0), dtype=torch.long)], dim=0)
        return x, y


def _extract_lines(split) -> List[str]:
    # Prefer common text columns; fall back conservatively
    if "sentence" in split.column_names:
        return split["sentence"]
    if "text" in split.column_names:
        return split["text"]
    # Fallback: join any string columns found
    for col in split.column_names:
        if isinstance(split[col][0], str):
            return split[col]
    return []


@register_dataset("ptb_text_only")
def build_ptb_text_only(max_samples: Optional[int], data_root: str) -> Tuple[object, object]:
    # Load PTB via generic text loader to avoid deprecated script datasets
    ds = load_dataset(
        "text",
        data_files={
            "train": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt",
            "validation": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt",
        },
        cache_dir=f"{data_root}/ptb_text_only",
    )
    train_lines = _extract_lines(ds["train"])
    val_lines = _extract_lines(ds["validation"])
    vocab = build_vocab_from_text(train_lines)
    train = PTBTokens(train_lines, vocab)
    val = PTBTokens(val_lines, vocab)
    train.vocab = vocab
    val.vocab = vocab
    return maybe_limit_split(train, val, max_samples)


