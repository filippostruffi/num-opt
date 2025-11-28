from typing import Any
import torch
from torch import nn
from benchmark.config.registry import register_model


class GRULanguageModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, num_layers: int = 2, dropout: float = 0.2, tie_weights: bool = True):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            # Weight tying expects same dimensionality
            self.lm_head.weight = self.tok_emb.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x T (token ids)
        h = self.tok_emb(x)
        h, _ = self.gru(h)
        h = self.dropout(h)
        return self.lm_head(h)


@register_model("gru_lm")
def build_model(task: Any, dataset: Any) -> nn.Module:
    vocab = getattr(dataset, "vocab", dataset.dataset.vocab)
    task.runtime_objects["vocab"] = vocab
    return GRULanguageModel(vocab_size=len(vocab))


