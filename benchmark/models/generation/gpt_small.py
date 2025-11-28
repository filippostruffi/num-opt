from typing import Any
import torch
from torch import nn
from benchmark.config.registry import register_model


class GPTSmall(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(512, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.size()
        pos = torch.arange(0, T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.tok_emb(x) + self.pos_emb(pos)
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        h = self.blocks(h, mask=mask)
        return self.lm_head(h)


@register_model("gpt_small")
def build_model(task: Any, dataset: Any) -> nn.Module:
    vocab = getattr(dataset, "vocab", dataset.dataset.vocab)
    task.runtime_objects["vocab"] = vocab
    return GPTSmall(vocab_size=len(vocab))


