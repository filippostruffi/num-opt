## Metrics

General
- Loss: task-specific loss (lower is better).
- Runtime/convergence: wall-clock to convergence/plateau (lower is better).
  - Definitions in logs:
    - total_runtime_s: training runtime for the optimizer run
    - epochs_to_converge/time_to_converge_s: first epoch/time achieving best primary metric
    - epochs_to_plateau/time_to_plateau_s: first epoch/time without improvement for patience

By Task
- Image Classification
  - Primary: Accuracy = correct / total
  - Range: [0,1], higher is better; small-sample runs often <0.7
  - Loss: Cross-entropy over logits
- Semantic Segmentation
  - Primary: Dice = mean_c 2·|Pred_c ∩ True_c| / (|Pred_c| + |True_c|)
  - Range: [0,1], higher is better; small-sample runs ~0.3–0.6
  - Loss: Cross-entropy with ignore_index=255
- Sentiment Analysis
  - Primary: Accuracy; Range ~0.5–0.8 on tiny samples
  - Loss: Cross-entropy (token-classification or HF loss)
- Machine Translation
  - Primary: BLEU (corpus-level n-gram precision with brevity penalty)
  - BLEU ∈ [0,100]; higher is better; tiny samples can be very low (<10)
  - Loss: Cross-entropy on next-token over target vocab
  - Non-primary:
    - ROUGE-L (LCS F1 proxy here): fluency/overlap sensitivity; higher is better
    - METEOR (proxy = ROUGE-L): synonym/recall-aware overlap; higher is better
    - TER (proxy = 1 − ROUGE-L): edit distance flavor; lower is better
- NER
  - Primary: F1 (micro over tags ignoring padding)
  - F1 ∈ [0,1]; small-sample F1 often <0.6; mature baselines on full data >0.9
  - Loss: CRF negative log-likelihood (learned transitions)
  - Non-primary:
    - Precision/Recall (micro over tags): trade-off insight (higher is better)
- Text Generation (Language Modeling)
  - Primary: Perplexity = exp(mean negative log-likelihood)
  - PPL ∈ [1,∞); lower is better; word-level small corpora often 50–200 early
  - Loss: Cross-entropy on next-token over vocab
  - Non-primary:
    - BLEU (proxy): overlap of generated vs target blocks (higher is better)
    - ROUGE-L (proxy): LCS-based similarity (higher is better)
    - METEOR (proxy = ROUGE-L): overlap sensitivity (higher is better)
- Text Summarization
  - Primary: ROUGE (R-1, R-2, R-L; we report an aggregate)
  - Higher is better; tiny-sample runs are modest (often <20)
  - Loss: Cross-entropy on decoder outputs (teacher forcing)
  - Non-primary:
    - BLEU: n-gram precision (higher is better)
    - METEOR (proxy = ROUGE-L): recall-friendly overlap (higher is better)
    - Compression ratio: len(candidate)/len(reference) (≈1 is typical; task-dependent)
- Question Answering
  - Primary: F1 (token overlap between predicted span and gold span)
  - F1 ∈ [0,1]; small-sample runs can be volatile due to exact match reliance
  - Loss: Sum of cross-entropy for start and end positions
  - Non-primary:
    - EM (Exact Match): 1 if predicted span equals gold span exactly (higher is better)
    - Precision/Recall: token-level overlap precision and recall (higher is better)

Why these
- They are standard, directly reflect prediction quality, and correlate with training stability/convergence under different optimizers.

Computation (high-level)
- Accuracy: mean(1{argmax(pred)==label}).
- Dice: mean_c 2·|Pred_c ∩ True_c| / (|Pred_c| + |True_c|).
- BLEU: brevity-penalized geometric mean of n-gram precisions.
- F1 (NER/QA): 2·precision·recall / (precision + recall).
- Perplexity: exp(average negative log-likelihood).
- ROUGE: token/n-gram/LCS overlap between generated and reference texts.

Interpretation tips
- Trust trends over absolute values on tiny samples (e.g., `--max-samples 2`)
- Optimizers that lower loss faster without degrading primary metrics are preferable
- For dense/seq2seq tasks, variance between runs (seeds) can be larger; consider averaging


