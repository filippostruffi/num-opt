from typing import Tuple, Optional, Dict, List, Any
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
        if context_tokens[i : i + m] == answer_tokens and m > 0:
            return i, i + m - 1
    return 0, 0


def _extract_fields(item: Dict[str, Any]) -> Tuple[str, str, str]:
    # TweetQA variants may use capitalized keys
    context = item.get("Tweet", item.get("tweet", ""))
    question = item.get("Question", item.get("question", ""))
    answers = item.get("Answer", item.get("answers", []))
    if isinstance(answers, dict):
        texts = answers.get("text", [])
    elif isinstance(answers, list):
        texts = answers
    else:
        texts = []
    answer_text = texts[0] if len(texts) > 0 else ""
    return str(context), str(question), str(answer_text)


class TweetQADataset(Dataset):
    def __init__(self, hf_split, vocab: Dict[str, int], max_ctx_len: int = 256, max_q_len: int = 64):
        self.data = hf_split
        self.vocab = vocab
        self.max_ctx_len = max_ctx_len
        self.max_q_len = max_q_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context, question, answer_text = _extract_fields(item)
        ctx_tokens = simple_tokenize(context)[: self.max_ctx_len]
        q_tokens = simple_tokenize(question)[: self.max_q_len]
        ans_tokens = simple_tokenize(answer_text)
        start, end = find_answer_span(ctx_tokens, ans_tokens)
        ctx_ids = [self.vocab.get(t, 1) for t in ctx_tokens]
        q_ids = [self.vocab.get(t, 1) for t in q_tokens]
        if len(ctx_ids) < self.max_ctx_len:
            ctx_ids = ctx_ids + [0] * (self.max_ctx_len - len(ctx_ids))
        if len(q_ids) < self.max_q_len:
            q_ids = q_ids + [0] * (self.max_q_len - len(q_ids))
        return {
            "context_ids": torch.tensor(ctx_ids, dtype=torch.long),
            "question_ids": torch.tensor(q_ids, dtype=torch.long),
            "start": torch.tensor(start, dtype=torch.long),
            "end": torch.tensor(end, dtype=torch.long),
        }


@register_dataset("tweet_qa")
def build_tweet_qa(max_samples: Optional[int], data_root: str) -> Tuple[object, object]:
    ds = load_dataset("tweet_qa", cache_dir=f"{data_root}/tweet_qa")
    # build vocab from training tweets/questions
    train_contexts = ds["train"]["Tweet"] if "Tweet" in ds["train"].column_names else ds["train"]["tweet"]
    train_questions = ds["train"]["Question"] if "Question" in ds["train"].column_names else ds["train"]["question"]
    vocab = build_vocab(list(train_contexts)[:10000] + list(train_questions)[:10000])
    train = TweetQADataset(ds["train"], vocab)
    val_split = ds["validation"] if "validation" in ds else ds["test"]
    val = TweetQADataset(val_split, vocab)
    train.vocab = vocab
    val.vocab = vocab
    return maybe_limit_split(train, val, max_samples)


