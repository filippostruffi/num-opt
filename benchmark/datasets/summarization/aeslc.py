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
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def encode(text: str, vocab: Dict[str, int], max_len: int = 64) -> List[int]:
    ids = [vocab["<bos>"]] + [vocab.get(t, vocab["<unk>"]) for t in simple_tokenize(text)][: max_len - 2] + [vocab["<eos>"]]
    return ids


class AESLCDataset(Dataset):
    def __init__(self, hf_split, src_vocab: Dict[str, int], tgt_vocab: Dict[str, int], max_src_len: int = 128, max_tgt_len: int = 32):
        self.data = hf_split
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        body = item["email_body"]
        subject = item["subject_line"]
        src_ids = torch.tensor(encode(body, self.src_vocab, self.max_src_len), dtype=torch.long)
        tgt_ids = torch.tensor(encode(subject, self.tgt_vocab, self.max_tgt_len), dtype=torch.long)
        return {"src": src_ids, "tgt": tgt_ids}


@register_dataset("aeslc")
def build_aeslc(max_samples: Optional[int], data_root: str) -> Tuple[object, object]:
    ds = load_dataset("aeslc", cache_dir=f"{data_root}/aeslc")
    # Build small vocabs from a subset to keep it light
    src_samples = ds["train"]["email_body"][:5000]
    tgt_samples = ds["train"]["subject_line"][:5000]
    src_vocab = build_vocab(src_samples)
    tgt_vocab = build_vocab(tgt_samples)
    train = AESLCDataset(ds["train"], src_vocab, tgt_vocab)
    # prefer validation if available; otherwise use test
    val_split = ds["validation"] if "validation" in ds else ds["test"]
    val = AESLCDataset(val_split, src_vocab, tgt_vocab)
    # attach vocabs
    train.src_vocab = src_vocab
    train.tgt_vocab = tgt_vocab
    val.src_vocab = src_vocab
    val.tgt_vocab = tgt_vocab
    return maybe_limit_split(train, val, max_samples)


