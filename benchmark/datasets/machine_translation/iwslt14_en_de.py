from typing import Tuple, Optional, List, Dict
from torch.utils.data import Dataset
from benchmark.datasets.base_dataset import maybe_limit_split
from benchmark.config.registry import register_dataset
import torch
from datasets import load_dataset


def simple_tokenize(s: str) -> List[str]:
    return s.strip().lower().split()


def build_vocab(samples: List[str], max_size: int = 40000) -> Dict[str, int]:
    from collections import Counter
    counter = Counter()
    for s in samples:
        counter.update(simple_tokenize(s))
    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    for word, _ in counter.most_common():
        if len(vocab) >= max_size:
            break
        vocab[word] = len(vocab)
    return vocab


def encode(s: str, vocab: Dict[str, int], max_len: int = 64) -> List[int]:
    ids = [vocab["<bos>"]] + [vocab.get(t, vocab["<unk>"]) for t in simple_tokenize(s)][: max_len - 2] + [vocab["<eos>"]]
    return ids


class IWSLTDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], src_vocab: Dict[str, int], tgt_vocab: Dict[str, int], max_len: int = 64):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        en, de = self.pairs[idx]
        return {
            "src": torch.tensor(encode(en, self.src_vocab, self.max_len), dtype=torch.long),
            "tgt": torch.tensor(encode(de, self.tgt_vocab, self.max_len), dtype=torch.long),
        }


@register_dataset("iwslt14_en_de")
def build_iwslt(max_samples: Optional[int], data_root: str) -> Tuple[object, object]:
    # Load a small real en-de parallel corpus (opus_books). IWSLT2017 loading scripts are deprecated on HF.
    ds = load_dataset("opus_books", "de-en", cache_dir=f"{data_root}/iwslt14_en_de")
    train_split = ds["train"]
    val_split = ds["validation"] if "validation" in ds else (ds["test"] if "test" in ds else None)
    # Build pairs
    train_pairs = [(ex["translation"]["en"], ex["translation"]["de"]) for ex in train_split]
    if val_split is None:
        # create a small validation from the end of train if no split provided
        n = max(1000, min(5000, len(train_pairs) // 10))
        valid_pairs = train_pairs[-n:]
        train_pairs = train_pairs[:-n]
    else:
        valid_pairs = [(ex["translation"]["en"], ex["translation"]["de"]) for ex in val_split]
    src_samples = [en for en, _ in train_pairs[:10000]]
    tgt_samples = [de for _, de in train_pairs[:10000]]
    src_vocab = build_vocab(src_samples)
    tgt_vocab = build_vocab(tgt_samples)
    train = IWSLTDataset(train_pairs, src_vocab, tgt_vocab)
    val = IWSLTDataset(valid_pairs, src_vocab, tgt_vocab)
    train.src_vocab = src_vocab
    train.tgt_vocab = tgt_vocab
    val.src_vocab = src_vocab
    val.tgt_vocab = tgt_vocab
    return maybe_limit_split(train, val, max_samples)


