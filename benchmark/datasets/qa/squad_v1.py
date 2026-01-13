from typing import Tuple, Optional, Dict, List
from datasets import load_dataset
from torch.utils.data import Dataset
from benchmark.datasets.base_dataset import maybe_limit_split
from benchmark.config.registry import register_dataset
import torch
import re


def simple_tokenize(s: str) -> List[str]:
    # Lowercase to improve vocab coverage and span robustness
    return s.strip().lower().split()


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


def tokenize_with_spans(s: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Tokenize by whitespace using original string spans so we can map
    character offsets to token indices reliably.
    """
    tokens: List[str] = []
    spans: List[Tuple[int, int]] = []
    for m in re.finditer(r"\S+", s):
        tokens.append(s[m.start() : m.end()])
        spans.append((m.start(), m.end()))
    return tokens, spans


def map_char_span_to_token_span(
    ctx_spans: List[Tuple[int, int]], answer_start_char: int, answer_text: str
) -> Optional[Tuple[int, int]]:
    """
    Map a character-level span (from SQuAD's answer_start and answer_text)
    to token indices based on token spans.
    Returns None if mapping fails.
    """
    if answer_text is None:
        return None
    ans_len = len(answer_text)
    if ans_len == 0 or answer_start_char is None or answer_start_char < 0:
        return None
    ans_start = answer_start_char
    ans_end = ans_start + ans_len - 1

    start_tok_idx = None
    end_tok_idx = None
    # Find token whose span covers ans_start
    for i, (s, e) in enumerate(ctx_spans):
        if s <= ans_start < e:
            start_tok_idx = i
            break
    # Find token whose span covers ans_end
    for j in range(len(ctx_spans) - 1, -1, -1):
        s, e = ctx_spans[j]
        if s <= ans_end < e:
            end_tok_idx = j
            break
    if start_tok_idx is None or end_tok_idx is None or end_tok_idx < start_tok_idx:
        return None
    return start_tok_idx, end_tok_idx


class SquadDataset(Dataset):
    def __init__(self, hf_split, vocab: Dict[str, int], max_len: int = 384):
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
        answer_starts = item["answers"]["answer_start"]
        answer_text = answers[0] if len(answers) > 0 else ""
        answer_start_char = answer_starts[0] if len(answer_starts) > 0 else None

        # Tokenize with spans for mapping, then lowercase tokens for ids
        raw_ctx_tokens, ctx_spans = tokenize_with_spans(context)
        # Truncate token lists based on max_len
        raw_ctx_tokens = raw_ctx_tokens[: self.max_len]
        ctx_spans = ctx_spans[: self.max_len]
        # Compute token indices for the answer using character offsets
        mapped = map_char_span_to_token_span(ctx_spans, answer_start_char, answer_text)
        if mapped is None:
            start, end = 0, 0
        else:
            start, end = mapped

        # Prepare lowercased ids for model input
        ctx_tokens = [t.lower() for t in raw_ctx_tokens]
        q_tokens = simple_tokenize(question)[: 64]
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


