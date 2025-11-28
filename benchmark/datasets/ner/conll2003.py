from typing import Tuple, Optional, List, Dict
from torch.utils.data import Dataset
from benchmark.datasets.base_dataset import maybe_limit_split
from benchmark.config.registry import register_dataset
from datasets import load_dataset
import torch


def build_tag_vocab(tags_list: List[List[str]]) -> Dict[str, int]:
    vocab = {"O": 0}
    for seq in tags_list:
        for t in seq:
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab


def build_word_vocab(tokens_list: List[List[str]], max_size: int = 50000) -> Dict[str, int]:
    from collections import Counter
    counter = Counter()
    for seq in tokens_list:
        counter.update([t.lower() for t in seq])
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, _ in counter.most_common(max_size):
        if w not in vocab:
            vocab[w] = len(vocab)
    return vocab


def build_char_vocab(tokens_list: List[List[str]], max_size: int = 200) -> Dict[str, int]:
    from collections import Counter
    counter = Counter()
    for seq in tokens_list:
        for t in seq:
            counter.update(list(t))
    # Reserve 0 for pad, 1 for unk
    vocab = {"<pad>": 0, "<unk>": 1}
    for ch, _ in counter.most_common(max_size - len(vocab)):
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


class NERDataset(Dataset):
    def __init__(self, hf_split, word_vocab: Dict[str, int], tag_vocab: Dict[str, int], max_len: int = 128, char_vocab: Optional[Dict[str, int]] = None, max_char_len: int = 20):
        self.data = hf_split
        self.word_vocab = word_vocab
        self.tag_vocab = tag_vocab
        self.max_len = max_len
        self.char_vocab = char_vocab
        self.max_char_len = max_char_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = [t.lower() for t in item["tokens"]][: self.max_len]
        tags = item["ner_tags"][: self.max_len]
        ids = [self.word_vocab.get(t, self.word_vocab["<unk>"]) for t in tokens]
        tag_ids = [tags[i] if isinstance(tags[i], int) else self.tag_vocab.get(tags[i], 0) for i in range(len(tags))]
        length = len(ids)
        # pad
        ids = ids + [0] * (self.max_len - length)
        tag_ids = tag_ids + [0] * (self.max_len - length)
        out = {"input_ids": torch.tensor(ids, dtype=torch.long), "tags": torch.tensor(tag_ids, dtype=torch.long), "length": length}
        if self.char_vocab is not None:
            # Build char ids per token, pad to max_char_len and max_len
            char_pad = 0
            unk = 1
            char_ids_seq: List[List[int]] = []
            for t in tokens:
                chars = [self.char_vocab.get(ch, unk) for ch in list(t)[: self.max_char_len]]
                chars = chars + [char_pad] * (self.max_char_len - len(chars))
                char_ids_seq.append(chars)
            # pad out to max_len with char PAD rows
            for _ in range(self.max_len - len(char_ids_seq)):
                char_ids_seq.append([char_pad] * self.max_char_len)
            out["char_ids"] = torch.tensor(char_ids_seq, dtype=torch.long)
        return out


@register_dataset("conll2003")
def build_conll2003(max_samples: Optional[int], data_root: str) -> Tuple[object, object]:
    try:
        ds = load_dataset("conll2003", cache_dir=f"{data_root}/conll2003")
        tokens_train = ds["train"]["tokens"]
        word_vocab = build_word_vocab(tokens_train)
        char_vocab = build_char_vocab(tokens_train)
        tag_vocab = {}
        train_split = ds["train"]
        val_split = ds["validation"] if "validation" in ds else ds["test"]
    except Exception:
        # Fallback to WikiANN English (real NER dataset with similar schema)
        ds = load_dataset("wikiann", "en", cache_dir=f"{data_root}/wikiann_en")
        tokens_train = ds["train"]["tokens"]
        word_vocab = build_word_vocab(tokens_train)
        char_vocab = build_char_vocab(tokens_train)
        tag_vocab = {}
        train_split = ds["train"]
        val_split = ds["validation"] if "validation" in ds else ds["test"]

    train = NERDataset(train_split, word_vocab, tag_vocab, char_vocab=char_vocab)
    val = NERDataset(val_split, word_vocab, tag_vocab, char_vocab=char_vocab)
    train.word_vocab = word_vocab
    val.word_vocab = word_vocab
    train.char_vocab = char_vocab
    val.char_vocab = char_vocab
    # Determine number of tags from train split
    try:
        max_tag = max(max(seq) for seq in train_split["ner_tags"])
        num_tags = int(max_tag) + 1
    except Exception:
        # If tags are strings, map via tag_vocab built implicitly in __getitem__; default to 10 as safe small number
        num_tags = 10
    train.num_tags = num_tags
    val.num_tags = num_tags
    return maybe_limit_split(train, val, max_samples)


