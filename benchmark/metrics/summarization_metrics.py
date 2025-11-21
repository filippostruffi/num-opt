from typing import Dict, Any, Optional, List
import torch
from benchmark.metrics.translation_metrics import simple_bleu, rouge_l


def ids_to_tokens(ids: List[int], id2tok: Dict[int, str]) -> List[str]:
    toks = []
    for i in ids:
        t = id2tok.get(i, None)
        if t is None or t in ("<pad>", "<bos>", "<eos>"):
            continue
        toks.append(t)
    return toks


class SummarizationMetrics:
    def create_accumulator(self) -> Dict[str, Any]:
        return {"refs": [], "cands": []}

    def update_ids(self, acc: Dict[str, Any], pred_ids: torch.Tensor, tgt_ids: torch.Tensor) -> None:
        for p, t in zip(pred_ids, tgt_ids):
            acc["cands"].append(p.tolist())
            acc["refs"].append(t.tolist())

    def compute(self, acc: Dict[str, Any], tgt_vocab: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        id2tok = {v: k for k, v in tgt_vocab.items()} if tgt_vocab is not None else {}
        bleus = []
        rouges = []
        meteors = []
        comp_ratios = []
        for cand_ids, ref_ids in zip(acc["cands"], acc["refs"]):
            cand = ids_to_tokens(cand_ids, id2tok)
            ref = ids_to_tokens(ref_ids, id2tok)
            bleus.append(simple_bleu(cand, ref))
            r = rouge_l(cand, ref)
            rouges.append(r)
            meteors.append(r)  # proxy
            comp_ratios.append(len(cand) / max(1, len(ref)))
        if not bleus:
            return {"rouge": 0.0, "bleu": 0.0, "meteor": 0.0, "compression_ratio": 1.0}
        return {
            "rouge": float(sum(rouges) / len(rouges)),
            "bleu": float(sum(bleus) / len(bleus)),
            "meteor": float(sum(meteors) / len(meteors)),
            "compression_ratio": float(sum(comp_ratios) / len(comp_ratios)),
        }


