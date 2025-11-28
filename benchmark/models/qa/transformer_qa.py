from typing import Any
import torch
from torch import nn
from benchmark.config.registry import register_model


class TransformerQA(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 4, num_layers: int = 2, ctx_len: int = 256, q_len: int = 64):
        super().__init__()
        self.ctx_len = ctx_len
        self.q_len = q_len
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_ctx = nn.Embedding(ctx_len, d_model)
        self.pos_q = nn.Embedding(q_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.ctx_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.q_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model * 2, d_model)
        self.start_head = nn.Linear(d_model, 1)
        self.end_head = nn.Linear(d_model, 1)

    def forward(self, context_ids, question_ids):
        B, Tc = context_ids.size()
        _, Tq = question_ids.size()
        pos_c = torch.arange(0, Tc, device=context_ids.device).unsqueeze(0).expand(B, Tc)
        pos_q = torch.arange(0, Tq, device=context_ids.device).unsqueeze(0).expand(B, Tq)
        ctx = self.emb(context_ids) + self.pos_ctx(pos_c)
        q = self.emb(question_ids) + self.pos_q(pos_q)
        ctx = self.ctx_encoder(ctx)
        q = self.q_encoder(q)
        # simple fusion: concat mean pooled q to each ctx position
        q_mean = q.mean(dim=1, keepdim=True).expand_as(ctx)
        h = torch.tanh(self.proj(torch.cat([ctx, q_mean], dim=-1)))
        start_logits = self.start_head(h).squeeze(-1)
        end_logits = self.end_head(h).squeeze(-1)
        return start_logits, end_logits


@register_model("transformer_qa")
def build_model(task: Any, dataset: Any) -> nn.Module:
    vocab = getattr(dataset, "vocab", dataset.dataset.vocab)
    task.runtime_objects["vocab"] = vocab
    return TransformerQA(vocab_size=len(vocab))


