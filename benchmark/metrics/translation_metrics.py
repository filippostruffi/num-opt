from typing import Dict, Any, List, Optional
import torch


def ids_to_tokens(ids: List[int], id2tok: Dict[int, str]) -> List[str]:
    toks = []
    for i in ids:
        if i in id2tok:
            token = id2tok[i]
            if token in ("<pad>", "<bos>", "<eos>"):
                continue
            toks.append(token)
    return toks


def simple_bleu(candidate: List[str], reference: List[str], max_n: int = 4) -> float:
    """
    Smoothed sentence BLEU with brevity penalty.
    - Smoothing: add-one smoothing on n-gram precision, i.e., (match+1)/(total+1)
    - Brevity penalty: exp(1 - r/c) if c < r else 1, where c=len(candidate), r=len(reference)
    Returns BLEU in [0, 1].
    """
    import math
    c_len = len(candidate)
    r_len = len(reference)
    if c_len == 0:
        return 0.0
    weights = [1.0 / max_n] * max_n
    log_prec_sum = 0.0
    for n in range(1, max_n + 1):
        c_ngrams = {}
        r_ngrams = {}
        # candidate n-grams
        for i in range(max(0, c_len - n + 1)):
            key = tuple(candidate[i : i + n])
            c_ngrams[key] = c_ngrams.get(key, 0) + 1
        # reference n-grams
        for i in range(max(0, r_len - n + 1)):
            key = tuple(reference[i : i + n])
            r_ngrams[key] = r_ngrams.get(key, 0) + 1
        match = 0
        total = max(1, c_len - n + 1)
        for k, v in c_ngrams.items():
            match += min(v, r_ngrams.get(k, 0))
        # add-one smoothing to avoid log(0)
        prec = (match + 1.0) / (total + 1.0)
        log_prec_sum += weights[n - 1] * math.log(max(prec, 1e-12))
    # brevity penalty
    if c_len < r_len:
        bp = math.exp(1.0 - float(r_len) / float(c_len))
    else:
        bp = 1.0
    return float(bp * math.exp(log_prec_sum))


def rouge_l(candidate: List[str], reference: List[str]) -> float:
    # LCS-based
    m, n = len(candidate), len(reference)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if candidate[i] == reference[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    lcs = dp[0][0]
    prec = lcs / max(1, m)
    rec = lcs / max(1, n)
    if (prec + rec) == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


class TranslationMetrics:
    def create_accumulator(self) -> Dict[str, Any]:
        return {"refs": [], "cands": []}

    def update_ids(self, acc: Dict[str, Any], pred_ids: torch.Tensor, tgt_ids: torch.Tensor) -> None:
        # pred_ids, tgt_ids: BxT
        for p, t in zip(pred_ids, tgt_ids):
            acc["cands"].append(p.tolist())
            acc["refs"].append(t.tolist())

    def compute(self, acc: Dict[str, Any], tgt_vocab: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        if tgt_vocab is None:
            # build id2tok placeholder
            id2tok = {}
        else:
            id2tok = {v: k for k, v in tgt_vocab.items()}
        bleus = []
        rouges = []
        ters = []
        for cand_ids, ref_ids in zip(acc["cands"], acc["refs"]):
            cand = ids_to_tokens(cand_ids, id2tok)
            ref = ids_to_tokens(ref_ids, id2tok)
            bleus.append(simple_bleu(cand, ref))
            rouges.append(rouge_l(cand, ref))
            # TER proxy: 1 - ROUGE_L
            ters.append(1.0 - rouges[-1])
        if len(bleus) == 0:
            return {"bleu": 0.0, "rouge": 0.0, "ter": 1.0}
        return {
            "bleu": float(sum(bleus) / len(bleus)),
            "rouge": float(sum(rouges) / len(rouges)),
            "ter": float(sum(ters) / len(ters)),
        }


