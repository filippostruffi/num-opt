## Models (by task)

- Image Classification
  - cnn_scratch
    - Arch: Conv–ReLU–Pool stacks + linear head (few layers)
    - Params: ~1–2M (depends on channels)
    - Training: No pretraining; sensitive to LR and WD; good for optimizer separation
  - resnet18
    - Arch: Residual blocks with BN; downsampling via stride
    - Params: ~11M
    - Training: Stable; BN requires batch size >1; WD and momentum impactful

- Semantic Segmentation
  - unet
    - Arch: Encoder–decoder with skip connections; conv blocks + upsampling
    - Params: ~7–30M (config-dependent)
    - Training: Dice/CE possible; here CE; benefits from larger batches for BN stability
  - deeplabv3_resnet50
    - Arch: ResNet-50 backbone + ASPP head
    - Params: ~40–45M
    - Training: Robust dense predictor; LR scheduling and WD matter

- Sentiment Analysis
  - lstm_sentiment
    - Arch: Embedding → BiLSTM/GRU → pooled → linear
    - Params: ~1–5M
    - Training: Very fast; sensitive to max_len and embedding init; Adam/AdamW common
  - distilbert_sentiment
    - Arch: HF DistilBERT base + classification head
    - Params: ~66M
    - Training: Finetuning; AdamW with small LR; WD and warmup critical

- Machine Translation
  - lstm_seq2seq
    - Arch: BiLSTM encoder + LSTM decoder + attention
    - Params: ~10–40M
    - Training: Teacher-forcing; gradient clipping often helpful; Adam variants preferred
  - transformer_seq2seq
    - Arch: Transformer encoder–decoder (multi-head attention, FFN)
    - Params: ~30–60M (small configs)
    - Training: LR schedules help; AdamW/LAMB often effective

- Named Entity Recognition
  - lstm_crf_ner
    - Arch: Embedding → BiLSTM → linear → CRF (train with NLL)
    - Params: ~1–3M
    - Training: CRF transitions learned; optimizer affects emission calibration
  - bilstm_crf_charcnn_ner
    - Arch: + char-CNN features concatenated to word embeddings → BiLSTM → CRF
    - Params: +0.1–0.5M for char CNN depending on filters
    - Training: Often boosts rare/OOOV tokens; Adam/AdaBelief effective

- Text Generation (Language Modeling)
  - gpt_small
    - Arch: Token + positional embeddings → TransformerEncoder with causal mask → LM head
    - Params: ~5–10M (small)
    - Training: Perplexity-sensitive; AdamW/SAM variants show differences
  - gru_lm
    - Arch: Embedding → 2-layer GRU → dropout → tied linear head
    - Params: ~3–6M
    - Training: Very fast; stable on small corpora; good counterpoint to attention

- Text Summarization
  - bart_small
    - Arch: Transformer (encoder–decoder) + sinusoidal pos encodings
    - Params: ~30M (small)
    - Training: Teacher-forcing; ROUGE sensitive to tokenization; AdamW baseline
  - tiny_transformer_seq2seq
    - Arch: Pre-LN Transformer encoder–decoder; tied target embeddings
    - Params: ~20–30M (with d_model=256, 3 layers)
    - Training: Lightweight, optimizer-sensitive; good for quick sweeps

- Question Answering
  - transformer_qa
    - Arch: Separate Transformer encoders for context and question; mean-pooled q fused into ctx; span heads
    - Params: ~10–20M
    - Training: Start/end CE; LR and WD influence span sharpness
  - bilstm_attention_qa
    - Arch: BiLSTM encoders; dot attention over question per context token; fusion + span heads
    - Params: ~5–10M
    - Training: Fast, robust; attention sharpness reflects optimizer step quality

Why these: each task has a simple and a stronger model to reveal optimizer differences under different architectures (RNN vs Transformer; simple vs CRF/attention; small vs stronger backbones).


