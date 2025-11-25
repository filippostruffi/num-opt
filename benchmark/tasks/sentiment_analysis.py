from typing import Dict, Any, Tuple, List
import torch
from torch import nn
from benchmark.tasks.base_task import BaseTask
from benchmark.metrics.sentiment import SentimentMetrics
from benchmark.config.registry import register_task


def text_to_ids(texts: List[str], vocab: Dict[str, int], max_len: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    ids = []
    lengths = []
    for t in texts:
        toks = t.strip().split()
        seq = [vocab.get(tok, vocab["<unk>"]) for tok in toks][:max_len]
        lengths.append(len(seq))
        seq = seq + [vocab["<pad>"]] * (max_len - len(seq))
        ids.append(seq)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


@register_task("sentiment_analysis")
class SentimentTask(BaseTask):
    @staticmethod
    def task_name() -> str:
        return "sentiment_analysis"

    @staticmethod
    def primary_metric_key() -> str:
        return "accuracy"

    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.metric = SentimentMetrics()
        self.runtime_objects: Dict[str, Any] = {}

    def create_metric_accumulator(self) -> Dict[str, Any]:
        return self.metric.create_accumulator()

    def update_metrics(self, acc: Dict[str, Any], outputs: torch.Tensor, targets: torch.Tensor) -> None:
        preds = torch.argmax(outputs, dim=1)
        self.metric.update(acc, preds.cpu(), targets.cpu())

    def compute_metrics(self, acc: Dict[str, Any]) -> Dict[str, Any]:
        return self.metric.compute(acc)

    def collate_fn(self, batch):
        # If a HF tokenizer is present (e.g., distilbert_sentiment), use it
        if "hf_tokenizer" in self.runtime_objects:
            tokenizer = self.runtime_objects["hf_tokenizer"]
            texts = [b[0] for b in batch]
            labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
            enc = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            enc = {k: v for k, v in enc.items()}
            enc["labels"] = labels
            return enc
        # Fallback: LSTM path with our simple vocab
        texts = [b[0] for b in batch]
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        vocab = self.runtime_objects["vocab"]
        ids, lengths = text_to_ids(texts, vocab)
        return {"input_ids": ids, "lengths": lengths, "labels": labels}

    def train_step(self, model: nn.Module, batch: Dict[str, Any]):
        if "hf_tokenizer" in self.runtime_objects:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask", None),
                labels=batch["labels"],
            )
            loss = outputs.loss
            logits = outputs.logits
            return loss, logits, batch["labels"]
        # LSTM path
        logits = model(batch["input_ids"], batch["lengths"])
        loss = self.criterion(logits, batch["labels"])
        return loss, logits, batch["labels"]

    def eval_step(self, model: nn.Module, batch: Dict[str, Any]):
        if "hf_tokenizer" in self.runtime_objects:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask", None),
                labels=batch["labels"],
            )
            loss = outputs.loss
            logits = outputs.logits
            return loss, logits, batch["labels"]
        # LSTM path
        logits = model(batch["input_ids"], batch["lengths"])
        loss = self.criterion(logits, batch["labels"])
        return loss, logits, batch["labels"]


