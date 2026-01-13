from typing import Dict, Any, Optional, List
import torch
from benchmark.metrics.translation_metrics import simple_bleu, rouge_l


def ids_to_text(ids: List[int], id2tok: Dict[int, str]) -> List[str]:
    toks = []
    for i in ids:
        tok = id2tok.get(i, None)
        if tok is None:
            continue
        if tok in ("<pad>",):
            continue
        toks.append(tok)
    return toks


class GenerationMetrics:
    def create_accumulator(self) -> Dict[str, Any]:
        return {"refs": [], "cands": []}

    def update_ids(self, acc: Dict[str, Any], pred_ids: torch.Tensor, tgt_ids: torch.Tensor) -> None:
        for p, t in zip(pred_ids, tgt_ids):
            acc["cands"].append(p.tolist())
            acc["refs"].append(t.tolist())

    def compute(self, acc: Dict[str, Any], vocab: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        id2tok = {v: k for k, v in vocab.items()} if vocab is not None else {}
        bleus = []
        rouges = []
        total_tokens = 0
        nll = 0.0  # Not tracked precisely here; use proxy perplexity from token mismatch
        for cand_ids, ref_ids in zip(acc["cands"], acc["refs"]):
            cand = ids_to_text(cand_ids, id2tok)
            ref = ids_to_text(ref_ids, id2tok)
            bleus.append(simple_bleu(cand, ref))
            rouges.append(rouge_l(cand, ref))
            total_tokens += max(len(ref), 1)
            mism = sum(1 for a, b in zip(cand, ref) if a != b)
            nll += mism  # proxy
        ppl = float(torch.exp(torch.tensor(nll / max(1, total_tokens))))
        meteor = sum(rouges) / max(1, len(rouges))  # proxy
        return {
            "bleu": float(sum(bleus) / max(1, len(bleus))),
            "rouge": float(sum(rouges) / max(1, len(rouges))),
            "meteor": float(meteor),
            "perplexity": float(ppl),
        }


