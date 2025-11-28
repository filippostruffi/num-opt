from typing import Any, Dict
import torch
from torch import nn
from benchmark.config.registry import register_model


class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 256, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt_in):
        src = self.pos_enc(self.src_emb(src))
        tgt = self.pos_enc(self.tgt_emb(tgt_in))
        out = self.transformer(src, tgt)
        return self.fc_out(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


@register_model("transformer_seq2seq")
def build_model(task: Any, dataset: Any) -> nn.Module:
    # dataset should carry src_vocab and tgt_vocab
    src_vocab_size = len(getattr(dataset, "src_vocab", dataset.dataset.src_vocab))
    tgt_vocab_size = len(getattr(dataset, "tgt_vocab", dataset.dataset.tgt_vocab))
    task.runtime_objects["src_vocab"] = getattr(dataset, "src_vocab", dataset.dataset.src_vocab)
    task.runtime_objects["tgt_vocab"] = getattr(dataset, "tgt_vocab", dataset.dataset.tgt_vocab)
    return TransformerSeq2Seq(src_vocab_size, tgt_vocab_size)


