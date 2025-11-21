from typing import Tuple, Optional, List, Dict
from torch.utils.data import Dataset
from benchmark.datasets.base_dataset import maybe_limit_split
from benchmark.config.registry import register_dataset
from datasets import load_dataset
import torch


def simple_tokenize(s: str) -> List[str]:
    return s.strip().lower().split()


def build_vocab(samples: List[str], max_size: int = 30000, min_freq: int = 1) -> Dict[str, int]:
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


def encode(text: str, vocab: Dict[str, int], max_len: int = 64) -> List[int]:
    ids = [vocab["<bos>"]]
    ids += [vocab.get(tok, vocab["<unk>"]) for tok in simple_tokenize(text)]
    ids = ids[: max_len - 1]
    ids.append(vocab["<eos>"])
    return ids


class MTDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], src_vocab: Dict[str, int], tgt_vocab: Dict[str, int], max_len: int = 64):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        en, de = self.pairs[idx]
        src_ids = torch.tensor(encode(en, self.src_vocab, self.max_len), dtype=torch.long)
        tgt_ids = torch.tensor(encode(de, self.tgt_vocab, self.max_len), dtype=torch.long)
        return {"src": src_ids, "tgt": tgt_ids}


@register_dataset("europarl_bilingual")
def build_europarl_bilingual(max_samples: Optional[int], data_root: str) -> Tuple[object, object]:
    # Use europarl_bilingual (de-en). If no validation/test split, derive a small validation from train.
    ds = load_dataset("europarl_bilingual", "de-en", cache_dir=f"{data_root}/europarl_bilingual")
    def to_pairs(split):
        return [(ex["translation"]["en"], ex["translation"]["de"]) for ex in split]
    train_pairs = to_pairs(ds["train"])
    if "validation" in ds:
        valid_pairs = to_pairs(ds["validation"])
    elif "test" in ds:
        valid_pairs = to_pairs(ds["test"])
    else:
        n = max(1000, min(5000, len(train_pairs) // 10))
        valid_pairs = train_pairs[-n:]
        train_pairs = train_pairs[:-n]
    # Build vocabs from a subset of training
    src_samples = [en for en, _ in train_pairs[:10000]]
    tgt_samples = [de for _, de in train_pairs[:10000]]
    src_vocab = build_vocab(src_samples)
    tgt_vocab = build_vocab(tgt_samples)
    train = MTDataset(train_pairs, src_vocab, tgt_vocab)
    val = MTDataset(valid_pairs, src_vocab, tgt_vocab)
    train.src_vocab = src_vocab
    train.tgt_vocab = tgt_vocab
    val.src_vocab = src_vocab
    val.tgt_vocab = tgt_vocab
    return maybe_limit_split(train, val, max_samples)


