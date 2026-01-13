from typing import Tuple, Optional, List, Dict
from datasets import load_dataset
from torch.utils.data import Dataset
from benchmark.datasets.base_dataset import maybe_limit_split
from benchmark.config.registry import register_dataset
import torch


def simple_tokenize(s: str) -> List[str]:
    return s.strip().lower().split()


def build_vocab(samples: List[str], max_size: int = 50000, min_freq: int = 1) -> Dict[str, int]:
    from collections import Counter
    counter = Counter()
    for s in samples:
        counter.update(simple_tokenize(s))
    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    for word, freq in counter.most_common():
        if freq < min_freq:
            continue
        if len(vocab) >= max_size:
            break
        vocab[word] = len(vocab)
    return vocab


def encode(text: str, vocab: Dict[str, int], max_len: int = 128) -> List[int]:
    ids = [vocab["<bos>"]] + [vocab.get(t, vocab["<unk>"]) for t in simple_tokenize(text)][: max_len - 2] + [vocab["<eos>"]]
    return ids


class CNNDailyMailDataset(Dataset):
    def __init__(self, hf_split, src_vocab: Dict[str, int], tgt_vocab: Dict[str, int], max_len: int = 128):
        self.data = hf_split
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        article = item["article"]
        summary = item["highlights"]
        src_ids = torch.tensor(encode(article, self.src_vocab, self.max_len), dtype=torch.long)
        tgt_ids = torch.tensor(encode(summary, self.tgt_vocab, self.max_len), dtype=torch.long)
        return {"src": src_ids, "tgt": tgt_ids}


@register_dataset("cnn_dailymail")
def build_cnn_dm(max_samples: Optional[int], data_root: str) -> Tuple[object, object]:
    ds = load_dataset("cnn_dailymail", "3.0.0", cache_dir=f"{data_root}/cnn_dailymail")
    # Build vocab from subset of train to limit size
    src_samples = ds["train"]["article"][:5000]
    tgt_samples = ds["train"]["highlights"][:5000]
    src_vocab = build_vocab(src_samples)
    tgt_vocab = build_vocab(tgt_samples)
    train = CNNDailyMailDataset(ds["train"], src_vocab, tgt_vocab)
    val = CNNDailyMailDataset(ds["validation"], src_vocab, tgt_vocab)
    # attach vocabs
    train.src_vocab = src_vocab
    train.tgt_vocab = tgt_vocab
    val.src_vocab = src_vocab
    val.tgt_vocab = tgt_vocab
    return maybe_limit_split(train, val, max_samples)


