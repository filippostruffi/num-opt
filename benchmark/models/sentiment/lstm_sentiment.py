from typing import Any, List, Dict
import torch
from torch import nn
from benchmark.config.registry import register_model


def build_vocab(samples: List[str], max_size: int = 30000, min_freq: int = 1) -> Dict[str, int]:
    from collections import Counter
    counter = Counter()
    for s in samples:
        counter.update(s.strip().split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.most_common():
        if freq < min_freq:
            continue
        if len(vocab) >= max_size:
            break
        vocab[word] = len(vocab)
    return vocab


class LSTMSentiment(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int = 2, emb_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        emb = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, (h, c) = self.lstm(packed)
        h = h[-1]
        h = self.drop(h)
        return self.fc(h)


@register_model("lstm_sentiment")
def build_model(task: Any, dataset: Any) -> nn.Module:
    # Build a small vocab from the training set samples
    texts = []
    for i in range(min(10000, len(dataset))):
        t, y = dataset[i]
        texts.append(t)
    vocab = build_vocab(texts)
    task.runtime_objects["vocab"] = vocab
    return LSTMSentiment(vocab_size=len(vocab))


