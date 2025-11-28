from typing import Tuple, Optional, List, Dict
from torch.utils import data
import torch
from datasets import load_dataset
from benchmark.datasets.base_dataset import maybe_limit_split
from benchmark.config.registry import register_dataset


def build_word_vocab(tokens_list: List[List[str]], max_size: int = 80000) -> Dict[str, int]:
    from collections import Counter
    counter = Counter()
    for seq in tokens_list:
        counter.update([t.lower() for t in seq])
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, _ in counter.most_recent() if hasattr(counter, "most_recent") else counter.most_common():
        if len(vocab) >= max_size:
            break
        if w not in vocab:
            vocab[w] = len(vocab)
    return vocab


def build_char_vocab(tokens_list: List[List[str]], max_size: int = 200) -> Dict[str, int]:
    from collections import Counter
    counter = Counter()
    for seq in tokens_list:
        for t in seq:
            counter.update(list(t))
    vocab = {"<pad>": 0, "<unk>": 1}
    for ch, _ in counter.most_common(max_size - len(vocab)):
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


class WikiAnnNERDataset(data.Dataset):
    def __init__(self, hf_split, word_vocab: Dict[str, int], max_len: int = 128, char_vocab: Optional[Dict[str, int]] = None, max_char_len: int = 20):
        self.data = hf_split
        self.word_vocab = word_vocab
        self.max_len = max_len
        self.char_vocab = char_vocab
        self.max_char_len = max_char_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = [t.lower() for t in item["tokens"]][: self.max_len]
        tags = item["ner_tags"][: self.max_len]
        ids = [self.word_vocab.get(tok, self.word_vocab["<unk>"]) for tok in tokens]
        tag_ids = [int(t) for t in tags]
        length = len(ids)
        # pad to fixed length with 0
        pad_len = self.max_len - length
        if pad_len > 0:
            ids = ids + [0] * pad_len
            tag_ids = tag_ids + [0] * pad_len
        out = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "tags": torch.tensor(tag_ids, dtype=torch.long),
            "length": length,
        }
        if self.char_vocab is not None:
            char_pad = 0
            unk = 1
            char_ids_seq: List[List[int]] = []
            for t in tokens:
                chars = [self.char_vocab.get(ch, unk) for ch in list(t)[: self.max_char_len]]
                chars = chars + [char_pad] * (self.max_char_len - len(chars))
                char_ids_seq.append(chars)
            for _ in range(self.max_len - len(char_ids_seq)):
                char_ids_seq.append([char_pad] * self.max_char_len)
            out["char_ids"] = torch.tensor(char_ids_seq, dtype=torch.long)
        return out


@register_dataset("wikiann_en")
def build_wikiann_en(max_samples: Optional[int], data_root: str) -> Tuple[object, object]:
    ds = load_dataset("wikiann", "en", cache_dir=f"{data_root}/wikiann_en")
    tokens_train = ds["train"]["tokens"]
    word_vocab = build_word_vocab(tokens_list=tokens_train)
    char_vocab = build_char_vocab(tokens_list=tokens_train)
    train = WikiAnnNERDataset(hf_split=ds["train"], word_vocab=word_vocab, char_vocab=char_vocab)
    # prefer validation if available, fall back to test
    val_split = ds.get("validation") if isinstance(ds, dict) and "validation" in ds else ds["test"]
    val = WikiAnnNERDataset(hf_split=val_split, word_vocab=word_vocab, char_vocab=char_vocab)
    # annotate num_tags for model building
    max_tag_train = max(max(ex) for ex in ds["train"]["ner_tags"])
    max_tag_val = max(max(ex) for ex in val_split["ner_tags"])
    num_tags = max(max_tag_train, max_tag_val) + 1
    train.num_tags = num_tags
    val.num_tags = num_tags
    train.word_vocab = word_vocab
    val.word_vocab = word_vocab
    train.char_vocab = char_vocab
    val.char_vocab = char_vocab
    return maybe_limit_split(train, val, max_samples)


