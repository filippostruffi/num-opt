from typing import Any
import torch
from torch import nn
from benchmark.config.registry import register_model


class CRF(nn.Module):
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))  # from_i -> to_j
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # emissions: BxTxC, tags: BxT, mask: BxT (1=valid)
        log_denominator = self._compute_log_partition(emissions, mask)
        log_numerator = self._compute_joint_likelihood(emissions, tags, mask)
        return torch.mean(log_denominator - log_numerator)

    def _compute_log_partition(self, emissions, mask):
        B, T, C = emissions.shape
        log_prob = self.start_transitions + emissions[:, 0]
        for t in range(1, T):
            emit = emissions[:, t].unsqueeze(2)  # BxCx1
            transition = self.transitions.unsqueeze(0)  # 1xCxC
            score = log_prob.unsqueeze(2) + transition + emit  # BxCxC
            log_prob_next = torch.logsumexp(score, dim=1)  # BxC
            log_prob = torch.where(mask[:, t].unsqueeze(1).bool(), log_prob_next, log_prob)
        log_prob = log_prob + self.end_transitions
        return torch.logsumexp(log_prob, dim=1)

    def _compute_joint_likelihood(self, emissions, tags, mask):
        B, T, C = emissions.shape
        score = self.start_transitions[tags[:, 0]] + emissions[range(B), 0, tags[:, 0]]
        for t in range(1, T):
            trans = self.transitions[tags[:, t - 1], tags[:, t]]
            emit = emissions[range(B), t, tags[:, t]]
            score = score + torch.where(mask[:, t].bool(), trans + emit, torch.zeros_like(emit))
        score = score + self.end_transitions[tags[range(B), mask.sum(1).long() - 1]]
        return score

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Viterbi decode
        B, T, C = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        paths = []
        backpointers = []
        for t in range(1, T):
            next_score = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_score, best_tags = next_score.max(dim=1)
            score = torch.where(mask[:, t].unsqueeze(1).bool(), best_score + emissions[:, t], score)
            backpointers.append(best_tags)
        score = score + self.end_transitions
        best_last_tag = score.argmax(dim=1)
        # Backtrack
        best_paths = [best_last_tag]
        for bp in reversed(backpointers):
            best_last_tag = bp[range(B), best_last_tag]
            best_paths.insert(0, best_last_tag)
        best_paths = torch.stack(best_paths, dim=1)
        return best_paths


class LSTMCRFNER(nn.Module):
    def __init__(self, vocab_size: int, num_tags: int, emb_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_tags)
        self.crf = CRF(num_tags)

    def forward(self, input_ids, lengths, tags=None):
        emb = self.emb(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=input_ids.size(1))
        emissions = self.fc(out)
        mask = (input_ids != 0).long()
        if tags is not None:
            loss = self.crf(emissions, tags, mask)
            return loss, emissions, mask
        else:
            return None, emissions, mask


@register_model("lstm_crf_ner")
def build_model(task: Any, dataset: Any) -> nn.Module:
    vocab_size = len(getattr(dataset, "word_vocab", dataset.dataset.word_vocab))
    num_tags = getattr(dataset, "num_tags", dataset.dataset.num_tags)
    task.runtime_objects["word_vocab"] = getattr(dataset, "word_vocab", dataset.dataset.word_vocab)
    return LSTMCRFNER(vocab_size=vocab_size, num_tags=num_tags)


