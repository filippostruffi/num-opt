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
        # Embeddings
        ctx = self.emb(context_ids) + self.pos_ctx(pos_c)
        q = self.emb(question_ids) + self.pos_q(pos_q)
        # Build padding masks (True = pad)
        ctx_pad_mask = context_ids.eq(0)  # B x Tc
        q_pad_mask = question_ids.eq(0)   # B x Tq
        # Encode with key padding masks
        ctx = self.ctx_encoder(ctx, src_key_padding_mask=ctx_pad_mask)
        q = self.q_encoder(q, src_key_padding_mask=q_pad_mask)
        # Masked mean pooling over question (ignore pads)
        q_valid = (~q_pad_mask).float()                      # B x Tq (1 for valid)
        valid_counts = q_valid.sum(dim=1, keepdim=True).clamp_min(1.0)  # B x 1
        q_sum = (q * q_valid.unsqueeze(-1)).sum(dim=1)       # B x D
        q_mean = (q_sum / valid_counts).unsqueeze(1)         # B x 1 x D
        q_mean = q_mean.expand_as(ctx)                       # B x Tc x D
        # Fusion and heads
        h = torch.tanh(self.proj(torch.cat([ctx, q_mean], dim=-1)))
        start_logits = self.start_head(h).squeeze(-1)  # B x Tc
        end_logits = self.end_head(h).squeeze(-1)      # B x Tc
        # Prevent predicting padded positions
        neg_inf = torch.finfo(start_logits.dtype).min
        start_logits = start_logits.masked_fill(ctx_pad_mask, neg_inf)
        end_logits = end_logits.masked_fill(ctx_pad_mask, neg_inf)
        return start_logits, end_logits


@register_model("transformer_qa")
def build_model(task: Any, dataset: Any) -> nn.Module:
    base = getattr(dataset, "dataset", dataset)
    vocab = getattr(dataset, "vocab", getattr(base, "vocab", None))
    if vocab is None:
        raise ValueError("vocab not found on dataset")
    task.runtime_objects["vocab"] = vocab
    # Align model positional embeddings with dataset context/question lengths when available
    ctx_len = getattr(dataset, "max_len", getattr(base, "max_len", 256))
    q_len = 64
    return TransformerQA(vocab_size=len(vocab), ctx_len=ctx_len, q_len=q_len)


