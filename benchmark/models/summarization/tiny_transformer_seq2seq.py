from typing import Any
import torch
from torch import nn
from benchmark.config.registry import register_model


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 1 x L x D
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TinyTransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True, norm_first=True
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, tgt_vocab_size, bias=False)
        if tie_embeddings:
            if self.lm_head.weight.shape == self.tgt_emb.weight.shape:
                self.lm_head.weight = self.tgt_emb.weight

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        # src: B x S, tgt_in: B x T
        src_h = self.pos_enc(self.src_emb(src))
        tgt_h = self.pos_enc(self.tgt_emb(tgt_in))
        # Causal mask for decoder self-attention
        T = tgt_in.size(1)
        tgt_mask = torch.triu(torch.ones(T, T, device=tgt_in.device), diagonal=1).bool()
        mem = self.encoder(src_h)
        out = self.decoder(tgt_h, mem, tgt_mask=tgt_mask)
        return self.lm_head(out)


@register_model("tiny_transformer_seq2seq")
def build_model(task: Any, dataset: Any) -> nn.Module:
    src_vocab_size = len(getattr(dataset, "src_vocab", dataset.dataset.src_vocab))
    tgt_vocab_size = len(getattr(dataset, "tgt_vocab", dataset.dataset.tgt_vocab))
    task.runtime_objects["src_vocab"] = getattr(dataset, "src_vocab", dataset.dataset.src_vocab)
    task.runtime_objects["tgt_vocab"] = getattr(dataset, "tgt_vocab", dataset.dataset.tgt_vocab)
    return TinyTransformerSeq2Seq(src_vocab_size, tgt_vocab_size)


