from typing import Tuple, Optional, Dict, List
from datasets import load_dataset
from torch.utils.data import Dataset
from benchmark.datasets.base_dataset import maybe_limit_split
from benchmark.config.registry import register_dataset
import torch


def simple_tokenize(s: str) -> List[str]:
    return s.strip().split()


def build_vocab(samples: List[str], max_size: int = 80000) -> Dict[str, int]:
    from collections import Counter
    counter = Counter()
    for s in samples:
        counter.update(simple_tokenize(s))
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, _ in counter.most_common(max_size - len(vocab)):
        if w not in vocab:
            vocab[w] = len(vocab)
    return vocab


def find_answer_span(context_tokens: List[str], answer_tokens: List[str]) -> Tuple[int, int]:
    n, m = len(context_tokens), len(answer_tokens)
    for i in range(n - m + 1):
        if context_tokens[i : i + m] == answer_tokens:
            return i, i + m - 1
    return 0, 0


class SquadDataset(Dataset):
    def __init__(self, hf_split, vocab: Dict[str, int], max_len: int = 256):
        self.data = hf_split
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item["context"]
        question = item["question"]
        answers = item["answers"]["text"]
        answer_text = answers[0] if len(answers) > 0 else ""
        ctx_tokens = simple_tokenize(context)[: self.max_len]
        q_tokens = simple_tokenize(question)[: 64]
        ans_tokens = simple_tokenize(answer_text)
        start, end = find_answer_span(ctx_tokens, ans_tokens)
        ctx_ids = [self.vocab.get(t, 1) for t in ctx_tokens]
        q_ids = [self.vocab.get(t, 1) for t in q_tokens]
        ctx_ids = ctx_ids + [0] * (self.max_len - len(ctx_ids))
        q_ids = q_ids + [0] * (64 - len(q_ids))
        return {
            "context_ids": torch.tensor(ctx_ids, dtype=torch.long),
            "question_ids": torch.tensor(q_ids, dtype=torch.long),
            "start": torch.tensor(start, dtype=torch.long),
            "end": torch.tensor(end, dtype=torch.long),
        }


@register_dataset("squad_v1")
def build_squad_v1(max_samples: Optional[int], data_root: str) -> Tuple[object, object]:
    ds = load_dataset("squad", cache_dir=f"{data_root}/squad_v1")
    # build vocab from training contexts/questions
    train_contexts = ds["train"]["context"][:10000]
    train_questions = ds["train"]["question"][:10000]
    vocab = build_vocab(train_contexts + train_questions)
    train = SquadDataset(ds["train"], vocab)
    val = SquadDataset(ds["validation"], vocab)
    train.vocab = vocab
    val.vocab = vocab
    return maybe_limit_split(train, val, max_samples)


