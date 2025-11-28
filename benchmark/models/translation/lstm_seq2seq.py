from typing import Any, Tuple
import torch
from torch import nn
from benchmark.config.registry import register_model


@register_model("lstm_seq2seq")
def build_model(task: Any, dataset: Any) -> nn.Module:
    # dataset should carry src_vocab and tgt_vocab
    src_vocab = getattr(dataset, "src_vocab", dataset.dataset.src_vocab)
    tgt_vocab = getattr(dataset, "tgt_vocab", dataset.dataset.tgt_vocab)
    task.runtime_objects["src_vocab"] = src_vocab
    task.runtime_objects["tgt_vocab"] = tgt_vocab
    return LSTMSeq2Seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        emb_dim=256,
        enc_hidden=256,
        dec_hidden=256,
        num_layers=1,
        dropout=0.1,
        pad_idx=0,
    )


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden: int, num_layers: int, dropout: float, pad_idx: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(emb_dim, hidden, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.0 if num_layers == 1 else dropout)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # src: BxS
        x = self.dropout(self.emb(src))
        outputs, (h, c) = self.rnn(x)  # outputs: BxSx(2H)
        return outputs, (h, c)


class DotAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # query: BxH ; keys: BxSxH ; values: BxSxH ; mask: BxS (1 valid, 0 pad)
        # scores = (keys • query)
        scores = torch.bmm(keys, query.unsqueeze(2)).squeeze(2)  # BxS
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=1)  # BxS
        ctx = torch.bmm(attn.unsqueeze(1), values).squeeze(1)  # BxH
        return ctx


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden: int, enc_hidden: int, num_layers: int, dropout: float, pad_idx: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(emb_dim + enc_hidden, hidden, num_layers=num_layers, batch_first=True, dropout=0.0 if num_layers == 1 else dropout)
        self.attn = DotAttention()
        self.fc_out = nn.Linear(hidden + enc_hidden, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden
        self.enc_hidden = enc_hidden

    def forward_step(self, inp_tok: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor], enc_out: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # inp_tok: B ; hidden: (h,c) each LxBxH ; enc_out: BxSx(2Henc) we will project to Henc
        B, S, twoH = enc_out.size()
        Henc = twoH // 2
        # Combine bi-encoder outputs by sum
        keys = enc_out.view(B, S, 2, Henc).sum(dim=2)  # BxSxHenc
        values = keys
        # Query from last layer hidden
        h_t = hidden[0][-1]  # BxHdec
        # To align dims, ensure Hdec == Henc; if not, project query
        if h_t.size(1) != Henc:
            # simple linear projection to encoder hidden size
            if not hasattr(self, "_proj_q"):
                self._proj_q = nn.Linear(h_t.size(1), Henc).to(h_t.device)
            q = self._proj_q(h_t)
        else:
            q = h_t
        ctx = self.attn(q, keys, values, src_mask)  # BxHenc
        emb = self.dropout(self.emb(inp_tok))  # BxE
        rnn_in = torch.cat([emb, ctx], dim=1).unsqueeze(1)  # Bx1x(E+Henc)
        out, hidden = self.rnn(rnn_in, hidden)  # out: Bx1xHdec
        out = out.squeeze(1)  # BxHdec
        if out.size(1) != Henc:
            if not hasattr(self, "_proj_out"):
                self._proj_out = nn.Linear(out.size(1), Henc).to(out.device)
            out_proj = self._proj_out(out)
            dec_feat = out_proj
        else:
            dec_feat = out
        logits = self.fc_out(torch.cat([dec_feat, ctx], dim=1))  # BxV
        return logits, hidden

    def forward(self, tgt_in: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor], enc_out: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        B, T = tgt_in.size()
        logits = []
        inp = tgt_in[:, 0]  # first token (usually <bos>)
        h = hidden
        for t in range(T):
            logit_t, h = self.forward_step(inp, h, enc_out, src_mask)
            logits.append(logit_t.unsqueeze(1))
            if t + 1 < T:
                inp = tgt_in[:, t + 1]
        return torch.cat(logits, dim=1)  # BxTxV


class LSTMSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, emb_dim: int, enc_hidden: int, dec_hidden: int, num_layers: int, dropout: float, pad_idx: int):
        super().__init__()
        self.pad_idx = pad_idx
        self.encoder = Encoder(src_vocab_size, emb_dim, enc_hidden, num_layers, dropout, pad_idx)
        # Decoder attends over encoder outputs reduced to Henc; decoder hidden may differ, handled inside Decoder
        self.decoder = Decoder(tgt_vocab_size, emb_dim, dec_hidden, enc_hidden, num_layers, dropout, pad_idx)

        # Bridge to initialize decoder hidden from encoder hidden
        self.bridge_h = nn.Linear(enc_hidden * 2, dec_hidden)
        self.bridge_c = nn.Linear(enc_hidden * 2, dec_hidden)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        # Build source mask
        src_mask = (src != self.pad_idx).long()  # BxS
        enc_out, (h, c) = self.encoder(src)  # h,c: (2L)xBxHenc
        # Concatenate last layer forward/backward states
        h_last_f = h[-2]
        h_last_b = h[-1]
        c_last_f = c[-2]
        c_last_b = c[-1]
        h0 = torch.tanh(self.bridge_h(torch.cat([h_last_f, h_last_b], dim=1)))  # BxHdec
        c0 = torch.tanh(self.bridge_c(torch.cat([c_last_f, c_last_b], dim=1)))  # BxHdec
        # Expand to num_layers for decoder
        num_layers_dec = self.decoder.rnn.num_layers
        h0 = h0.unsqueeze(0).repeat(num_layers_dec, 1, 1).contiguous()
        c0 = c0.unsqueeze(0).repeat(num_layers_dec, 1, 1).contiguous()
        logits = self.decoder(tgt_in, (h0, c0), enc_out, src_mask)
        return logits


