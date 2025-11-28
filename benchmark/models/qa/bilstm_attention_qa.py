from typing import Any
import torch
from torch import nn
from benchmark.config.registry import register_model


class BiLSTMAttentionQA(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 256, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.ctx_lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.q_lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        # Projection after fusion (ctx, attended_q, ctx*attended_q)
        fuse_dim = hidden_dim * 2 * 3
        self.proj = nn.Linear(fuse_dim, hidden_dim * 2)
        self.start_head = nn.Linear(hidden_dim * 2, 1)
        self.end_head = nn.Linear(hidden_dim * 2, 1)

    @staticmethod
    def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # scores: B x Tc x Tq, mask: B x Tq (1=valid)
        mask = mask.unsqueeze(1).expand_as(scores)  # B x Tc x Tq
        scores = scores.masked_fill(~mask.bool(), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        return attn

    def forward(self, context_ids: torch.Tensor, question_ids: torch.Tensor):
        # Shapes
        B, Tc = context_ids.size()
        _, Tq = question_ids.size()
        # Embeddings
        ctx = self.emb(context_ids)  # B x Tc x E
        q = self.emb(question_ids)   # B x Tq x E
        # Encode
        ctx_out, _ = self.ctx_lstm(ctx)  # B x Tc x 2H
        q_out, _ = self.q_lstm(q)        # B x Tq x 2H
        ctx_out = self.dropout(ctx_out)
        q_out = self.dropout(q_out)
        # Attention: for each ctx position attend over question
        # Similarity: dot-product
        sim = torch.bmm(ctx_out, q_out.transpose(1, 2))  # B x Tc x Tq
        # Build question mask (non-zero tokens are valid)
        q_mask = question_ids.ne(0)  # B x Tq
        attn = self._masked_softmax(sim, q_mask)  # B x Tc x Tq
        attended_q = torch.bmm(attn, q_out)  # B x Tc x 2H
        # Fusion
        fused = torch.cat([ctx_out, attended_q, ctx_out * attended_q], dim=-1)  # B x Tc x 6H
        h = torch.tanh(self.proj(fused))  # B x Tc x 2H
        # Start/end logits per token
        start_logits = self.start_head(h).squeeze(-1)  # B x Tc
        end_logits = self.end_head(h).squeeze(-1)      # B x Tc
        return start_logits, end_logits


@register_model("bilstm_attention_qa")
def build_model(task: Any, dataset: Any) -> nn.Module:
    vocab = getattr(dataset, "vocab", dataset.dataset.vocab)
    task.runtime_objects["vocab"] = vocab
    return BiLSTMAttentionQA(vocab_size=len(vocab))


