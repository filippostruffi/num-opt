## Tasks (What and Why)

- Image Classification
  - Predict a single class per image. Ubiquitous benchmark for optimization stability and speed.
  - Loss: Cross-entropy. Primary metric: Accuracy.
  - Batching: Standard tensors (B×C×H×W, B labels). BN favors batch size >1.
  - Pitfalls: Without normalization/augmentation, may saturate quickly on small samples.

- Semantic Segmentation
  - Pixel-wise classification. Tests optimization under dense predictions and class imbalance.
  - Loss: Cross-entropy with ignore_index=255. Primary: Dice.
  - Batching: (B×C×H×W, B×H×W mask). We drop_last=True to keep BN stable.
  - Pitfalls: Class imbalance; tiny batch sizes destabilize BN/statistics.

- Sentiment Analysis
  - Text classification (positive/negative). Classic NLP task; quick, revealing for optimizer behavior.
  - Loss: Cross-entropy (HF models return built-in loss). Primary: Accuracy.
  - Batching: Either dict of tokenized tensors (HF) or our ids/lengths (LSTM).
  - Pitfalls: Tokenization choice affects results; max_len truncation matters.

- Machine Translation
  - Seq2seq generation from source to target language. Sensitive to optimizer/learning-rate choices.
  - Loss: Cross-entropy teacher-forcing (tgt_in→tgt_out). Primary: BLEU (computed from ids).
  - Batching: pad_seq to max length per batch; decoder consumes tgt_in; predicts tgt_out.
  - Pitfalls: Vocab quality (OOV) impacts BLEU; LR schedule improves Transformers.

- Named Entity Recognition (NER)
  - Sequence labeling with structured decoding (CRF). Evaluates optimizers in structured prediction.
  - Loss: CRF negative log-likelihood. Primary: F1 over tags.
  - Batching: dict with `input_ids`, `tags`, `lengths` (+ optional `char_ids`).
  - Pitfalls: Mask/length consistency is critical; char features help rare/OOOV tokens.

- Text Generation (Language Modeling)
  - Next-token prediction. Perplexity directly reflects optimization quality on generative models.
  - Loss: Cross-entropy on next token. Primary: Perplexity.
  - Batching: Fixed block size packing from concatenated token streams.
  - Pitfalls: Simple tokenization; block size influences difficulty (shorter = easier).

- Text Summarization
  - Seq2seq abstraction/compression. Good for optimizer comparison in encoder–decoder models.
  - Loss: Cross-entropy on decoder outputs (teacher-forcing). Primary: ROUGE.
  - Batching: `{"src", "tgt_in", "tgt_out"}` with padding; causal mask in decoder.
  - Pitfalls: Short targets (e.g., AESLC) can saturate quickly; consider label smoothing.

- Question Answering (extractive)
  - Span prediction from context. Combines encoding, attention/fusion, and span heads.
  - Loss: start CE + end CE. Primary: F1 (token overlap).
  - Batching: dict with `context_ids`, `question_ids`, `starts`, `ends`.
  - Pitfalls: Exact-match spans rely on consistent tokenization; short contexts (TweetQA) can be tricky.

These cover core supervised ML regimes (classification, sequence labeling, seq2seq generation, dense prediction) across vision and NLP, making them practical and representative for optimizer benchmarking. 


