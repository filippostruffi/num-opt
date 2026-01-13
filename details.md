### Overview
This document enumerates techniques you can apply across all task/dataset/model combinations in this repository to improve generalization and reduce overfitting. It is intentionally exhaustive so techniques are not overlooked. For each area we list:
- **What** the technique does  
- **Why** it helps  
- **Where** in this repo it typically applies (dataset transforms, task loss, model, or trainer)  
- **Defaults in this repo** and **how to adjust**

Note: Some items are already enabled by default; others are recommendations you can add (or extend) with small code changes. This guide groups options by task, then provides cross-cutting controls (schedulers, clipping, weight decay, etc.).

---

## Cross-cutting training controls

- **Best checkpointing**
  - What: Save model weights for the best primary validation metric.
  - Why: Restores the peak model even if later epochs degrade.
  - Where: Trainer saves `<run_root>/<optimizer>/best_model.pt` (implemented).

- **Learning-rate schedulers**
  - What: Adjust LR during training (cosine, ReduceLROnPlateau, etc.).
  - Why: Reduces overfitting by annealing LR; improves stability.
  - Where: `Trainer.fit` takes a scheduler (implemented). Auto-picks a sensible default per task.
  - Defaults: Auto: cosine for image/segmentation; plateau on val loss for NLP/seq tasks. Override with `--scheduler {cosine,plateau,none}`.
  - In these benchmark commands, schedulers are chosen to balance convergence quality with per-epoch runtime on an A100.

- **Gradient clipping**
  - What: Clip global gradient norm.
  - Why: Stabilizes RNN/Transformer training; curbs exploding gradients.
  - Where: `Trainer._train_one_epoch` (implemented).
  - Defaults: Auto: 1.0 for NLP/seq tasks, off for image (override with `--max-grad-norm`).
  - In the commands below, all sequence-like tasks explicitly set `--max-grad-norm 1.0`.

- **Weight decay (L2 regularization)**
  - What: Penalize large weights, typically decoupled (AdamW) or classic L2.
  - Why: Regularizes parameters, improving generalization.
  - Where: Param group `weight_decay` (set per optimizer in `main.py`).
  - Defaults: Heuristics per optimizer family (e.g., SGD 5e-4, AdamW 0.01). Adjust in `main.py` as needed.

- **Label smoothing**
  - What: Smooth one-hot targets.
  - Why: Reduces overconfidence; often improves validation metrics.
  - Where: Task loss construction (implemented where applicable).
  - Defaults: Image classification 0.1; Sentiment 0.05; MT/Summarization 0.1 (ignores pad tokens).

- **Dropout**
  - What: Randomly drop activations.
  - Why: Reduces co-adaptation; improves robustness.
  - Where: Model definitions.
  - Defaults: LSTM sentiment has 0.5 dropout (implemented). Add/adjust in other models as needed.

- **Sharpness-aware optimizers (SAM/GSAM)**
  - What: Optimize flat minima via neighborhood-aware gradient steps.
  - Why: Can improve generalization and robustness.
  - Where: `suite/optimizers/advanced.py` (SAM/GSAM available). Already supported in the training loop.
  - Notes: Use with `--optimizers sam` or `gsam`; tune `rho` in the optimizer implementation if required.

---

## Task-specific techniques

### 1) Image Classification  
(datasets: `mnist`, `cifar10`; models: `cnn_scratch`, `resnet18`)

- **Preprocessing and augmentation**
  - Normalize inputs (per dataset stats). Default: enabled.
  - Geometric/color augmentations:
    - MNIST: `RandomRotation(10)` (enabled by default).
    - CIFAR-10: `RandomCrop(32, pad=4)` + `RandomHorizontalFlip` (enabled by default).
    - Extensions you can add: `ColorJitter`, `RandomGrayscale`, Cutout; or policies like RandAugment/AutoAugment.
  - Mixup/CutMix (optional extension):
    - Why: Strong regularization by mixing samples/labels.
    - Where: Dataloader collate or training step. Not implemented by default; recommended for CIFAR-10.

- **Loss and regularization**
  - Label smoothing 0.1 (enabled by default).
  - Weight decay: 5e-4 for SGD; 0.01 for AdamW (pre-set in `main.py`).
  - Dropout: Add to fully-connected heads if needed.
  - Stochastic depth / DropPath (extension, for deeper nets).

- **Optimizers and LR**
  - Optimizers: SGD/SGD+momentum, Adam/AdamW, RMSProp, etc.
  - Scheduler: cosine annealing (default), or step decay, or ReduceLROnPlateau.
  - Warmup (extension): Add linear warmup for first few epochs if desired.

- **Monitoring and selection**
  - Primary metric: accuracy.

**Recommended run commands (20 epochs, ~sub-second–few-seconds/epoch on A100)**

```bash
# MNIST / CNN scratch — all optimizers, cosine LR, fast and representative
python -m benchmark.main \
  --task image_classification --dataset mnist --model cnn_scratch \
  --optimizers all \
  --epochs 20 --batch-size 128 --max-samples 20000 --num-workers 8
```

```bash
# MNIST / ResNet18 — all optimizers, cosine LR
python -m benchmark.main \
  --task image_classification --dataset mnist --model resnet18 \
  --optimizers all \
  --epochs 20 --batch-size 128 --max-samples 20000 --num-workers 8
```

```bash
# CIFAR-10 / CNN scratch — all optimizers, cosine LR, data aug enabled by default
python -m benchmark.main \
  --task image_classification --dataset cifar10 --model cnn_scratch \
  --optimizers all \
  --epochs 20 --batch-size 128 --max-samples 15000 --num-workers 8
```

```bash
# CIFAR-10 / ResNet18 — all optimizers, cosine LR, strong baseline
python -m benchmark.main \
  --task image_classification --dataset cifar10 --model resnet18 \
  --optimizers all \
  --epochs 20 --batch-size 128 --max-samples 15000 --num-workers 8
```

---

### 2) Sentiment Analysis  
(datasets: `sst2`, `imdb`; models: `lstm_sentiment`, `distilbert_sentiment`)

- **Preprocessing**
  - LSTM path: whitespace tokenization with PAD/UNK; fixed `max_len` (128). Vocab built from a subset of train (implemented).
  - HF model path: tokenizer with padding/truncation to fixed `max_len` (128).
  - Optional extensions: subword tokenization for LSTM; punctuation/URL normalization; lowercasing; stopword handling (usually not required for neural models).

- **Loss and regularization**
  - Label smoothing 0.05 (default).
  - LSTM dropout 0.5 between encoder and classifier (default).
  - Weight decay: 1e-5 to 1e-4 for RNNs, 0.01 for DistilBERT (AdamW style). Defaults set per optimizer.
  - Word dropout / embedding dropout (extension): randomly drop tokens/embeddings to regularize.

- **Optimizers and LR**
  - Adam/AdamW commonly best; SGD variants viable with tuned LR.
  - Scheduler: ReduceLROnPlateau (default), or cosine.
  - Gradient clipping: 1.0 (default).
  - For DistilBERT we provide explicit `--lr-overrides` tuned for Transformer-style finetuning so that each optimizer uses a sensible LR.

- **Monitoring and selection**
  - Primary metric: accuracy.

**Recommended run commands (20 epochs, ≤~tens of seconds/epoch on A100)**

```bash
# SST-2 / LSTM Sentiment — all optimizers, plateau scheduler, gradient clipping
python -m benchmark.main \
  --task sentiment_analysis --dataset sst2 --model lstm_sentiment \
  --optimizers all \
  --epochs 20 --batch-size 64 --max-samples 30000 --num-workers 8
```

```bash
# SST-2 / DistilBERT Sentiment — all optimizers, plateau scheduler, HF-style LR overrides
python -m benchmark.main \
  --task sentiment_analysis --dataset sst2 --model distilbert_sentiment \
  --optimizers all \
  --epochs 20 --batch-size 64 --max-samples 10000 --num-workers 8 \
  --lr-overrides "sgd=1e-3,sgd_momentum=1e-4,rmsprop=3e-5,adagrad=3e-5,adadelta=1e-3,adam=3e-5,adamw=3e-5,radam=3e-5,lion=1e-4,lars=3e-5,lamb=3e-5,adabelief=3e-5,yogi=3e-5,adafactor=5e-5,sam=2e-5,gsam=2e-5"
```

```bash
# IMDB / LSTM Sentiment — all optimizers, plateau scheduler, clipping
python -m benchmark.main \
  --task sentiment_analysis --dataset imdb --model lstm_sentiment \
  --optimizers all \
  --epochs 20 --batch-size 16 --max-samples 115000 --num-workers 8
```

```bash
# IMDB / DistilBERT Sentiment — all optimizers, plateau scheduler, HF-style LR overrides
python -m benchmark.main \
  --task sentiment_analysis --dataset imdb --model distilbert_sentiment \
  --optimizers all \
  --epochs 20 --batch-size 64 --max-samples 90000 --num-workers 8 \
  --lr-overrides "sgd=1e-3,sgd_momentum=1e-4,rmsprop=3e-5,adagrad=3e-5,adadelta=1e-3,adam=3e-5,adamw=3e-5,radam=3e-5,lion=1e-4,lars=3e-5,lamb=3e-5,adabelief=3e-5,yogi=3e-5,adafactor=5e-5,sam=2e-5,gsam=2e-5"
```

---

### 3) Machine Translation  
(datasets: `iwslt14_en_de`, `europarl_bilingual`; models: `transformer_seq2seq`, `lstm_seq2seq`)

- **Preprocessing**
  - Sequence padding; teacher forcing (`tgt_in = tgt[:-1]`, `tgt_out = tgt[1:]`) (implemented).
  - Optional extensions: Subword tokenization (BPE/WordPiece), vocabulary truncation, length-based bucketing.

- **Loss and regularization**
  - CrossEntropy with `ignore_index=pad` and label smoothing 0.1 (default).
  - Dropout: 0.1–0.3 in attention/FF/residuals (add in model).
  - Weight decay: AdamW 0.01 for Transformers; for LSTM 1e-4 typical.
  - R-Drop (extension): KL consistency loss across dropout passes.

- **Optimizers and LR**
  - AdamW for Transformers; Adam for LSTM seq2seq.
  - Scheduler: ReduceLROnPlateau (default) or warmup+inverse-sqrt/linear decay (extension).
  - Gradient clipping: 1.0 (default).

- **Monitoring and selection**
  - Primary metric: BLEU.

**Recommended run commands (20 epochs, A100-friendly)**

```bash
# IWSLT14 En-De / Transformer seq2seq — all optimizers, plateau scheduler, clipping
python -m benchmark.main \
  --task machine_translation --dataset iwslt14_en_de --model transformer_seq2seq \
  --optimizers all \
  --epochs 20 --batch-size 64 --max-samples 10000 --num-workers 8
```

```bash
# IWSLT14 En-De / LSTM seq2seq — all optimizers, plateau scheduler, clipping
python -m benchmark.main \
  --task machine_translation --dataset iwslt14_en_de --model lstm_seq2seq \
  --optimizers all \
  --epochs 20 --batch-size 64 --max-samples 10000 --num-workers 8
```

```bash
# Europarl / Transformer seq2seq — all optimizers, plateau scheduler, clipping
python -m benchmark.main \
  --task machine_translation --dataset europarl_bilingual --model transformer_seq2seq \
  --optimizers all \
  --epochs 20 --batch-size 64 --max-samples 30000 --num-workers 8
```

```bash
# Europarl / LSTM seq2seq — all optimizers, plateau scheduler, clipping
python -m benchmark.main \
  --task machine_translation --dataset europarl_bilingual --model lstm_seq2seq \
  --optimizers all \
  --epochs 20 --batch-size 64 --max-samples 30000 --num-workers 8
```

---

### 4) Text Summarization  
(datasets: `cnn_dailymail`, `aeslc`; models: `bart_small`, `tiny_transformer_seq2seq`)

- **Preprocessing**
  - Sequence padding; teacher forcing (implemented).
  - Optional: limit max target length, truncate long sources.

- **Loss and regularization**
  - CrossEntropy with `ignore_index=pad` + label smoothing 0.1 (default).
  - Dropout/attn_dropout 0.1–0.3 (configure in model).
  - Weight decay: AdamW 0.01 typical.
  - R-Drop (extension) for additional regularization.

- **Optimizers and LR**
  - AdamW typically; optional warmup + linear decay (extension).
  - ReduceLROnPlateau or cosine as alternatives.
  - Gradient clipping: 1.0 (default).

- **Monitoring and selection**
  - Primary metric: ROUGE (e.g., ROUGE-L).

**Recommended run commands (20 epochs, tuned for ≤~40s/epoch)**

```bash
# CNN/DailyMail / BART-small (toy transformer) — all optimizers, plateau scheduler
# max-samples reduced to keep BART epochs within the target runtime on A100
python -m benchmark.main \
  --task text_summarization --dataset cnn_dailymail --model bart_small \
  --optimizers all \
  --epochs 20 --batch-size 16 --max-samples 8000 --num-workers 8
```

```bash
# CNN/DailyMail / Tiny Transformer seq2seq — all optimizers, plateau scheduler
python -m benchmark.main \
  --task text_summarization --dataset cnn_dailymail --model tiny_transformer_seq2seq \
  --optimizers all \
  --epochs 20 --batch-size 32 --max-samples 10000 --num-workers 8
```

```bash
# AESLC / BART-small (toy transformer) — all optimizers, plateau scheduler
python -m benchmark.main \
  --task text_summarization --dataset aeslc --model bart_small \
  --optimizers all \
  --epochs 20 --batch-size 16 --max-samples 10000 --num-workers 8
```

```bash
# AESLC / Tiny Transformer seq2seq — all optimizers, plateau scheduler
python -m benchmark.main \
  --task text_summarization --dataset aeslc --model tiny_transformer_seq2seq \
  --optimizers all \
  --epochs 20 --batch-size 32 --max-samples 10000 --num-workers 8
```

---

### 5) Text Generation (Language Modeling)  
(datasets: `wikitext2`, `ptb_text_only`; models: `gpt_small`, `gru_lm`)

- **Preprocessing**
  - Next-token prediction with padding to batch max length (implemented).
  - Optional: subword tokenization; context length tuning; dynamic batching.

- **Loss and regularization**
  - CrossEntropy with `ignore_index=pad`.
  - Dropout in embeddings/attn/FF 0.1–0.5 (configure in model).
  - Weight decay: 1e-5–1e-4 typical; AdamW 0.01 for Transformer-like models.

- **Optimizers and LR**
  - Adam/AdamW common; cosine decay + warmup (extension) often useful.
  - Gradient clipping: 1.0 (default).

- **Monitoring and selection**
  - Primary metric: perplexity (lower is better).

**Recommended run commands (20 epochs, A100-friendly)**

```bash
# WikiText-2 / GPT-small — all optimizers, cosine scheduler
python -m benchmark.main \
  --task text_generation --dataset wikitext2 --model gpt_small \
  --optimizers all \
  --epochs 20 --batch-size 64 --max-samples 15000 --num-workers 8
```

```bash
# WikiText-2 / GRU LM — all optimizers, cosine scheduler
python -m benchmark.main \
  --task text_generation --dataset wikitext2 --model gru_lm \
  --optimizers all \
  --epochs 20 --batch-size 64 --max-samples 15000 --num-workers 8
```

```bash
# PTB (text-only) / GPT-small — all optimizers, cosine scheduler
python -m benchmark.main \
  --task text_generation --dataset ptb_text_only --model gpt_small \
  --optimizers all \
  --epochs 20 --batch-size 64 --max-samples 50000 --num-workers 8
```

```bash
# PTB (text-only) / GRU LM — all optimizers, cosine scheduler
python -m benchmark.main \
  --task text_generation --dataset ptb_text_only --model gru_lm \
  --optimizers all \
  --epochs 20 --batch-size 64 --max-samples 50000 --num-workers 8
```

---

### 6) Question Answering  
(datasets: `squad_v1`, `tweet_qa`; models: `transformer_qa`, `bilstm_attention_qa`)

- **Preprocessing**
  - Stack context/question ids; supervise start/end (implemented).
  - Optional: subword tokenization; stride-based long context handling (extension).

- **Loss and regularization**
  - Sum of CrossEntropy for start and end positions.
  - Dropout 0.1–0.3; weight decay 0.01 (AdamW for Transformers), 1e-4 for LSTM.
  - Label smoothing is generally not used for span QA (keep off).

- **Optimizers and LR**
  - AdamW with linear decay+warmup (extension) or ReduceLROnPlateau.
  - Gradient clipping: 1.0 (default).

- **Monitoring and selection**
  - Primary metric: F1 (and EM).

**Recommended run commands (20 epochs, tuned max-samples for runtime)**

```bash
# SQuAD v1 / Transformer QA — all optimizers, plateau scheduler
# max-samples reduced to keep epochs under the runtime budget on A100
python -m benchmark.main \
  --task question_answering --dataset squad_v1 --model transformer_qa \
  --optimizers all \
  --epochs 20 --batch-size 32 --max-samples 20000 --num-workers 8
```

```bash
# SQuAD v1 / BiLSTM+Attention — all optimizers, plateau scheduler
python -m benchmark.main \
  --task question_answering --dataset squad_v1 --model bilstm_attention_qa \
  --optimizers all \
  --epochs 20 --batch-size 64 --max-samples 20000 --num-workers 8
```

```bash
# TweetQA / Transformer QA — all optimizers, plateau scheduler
python -m benchmark.main \
  --task question_answering --dataset tweet_qa --model transformer_qa \
  --optimizers all \
  --epochs 20 --batch-size 32 --max-samples 8000 --num-workers 8
```

```bash
# TweetQA / BiLSTM+Attention — all optimizers, plateau scheduler
python -m benchmark.main \
  --task question_answering --dataset tweet_qa --model bilstm_attention_qa \
  --optimizers all \
  --epochs 20 --batch-size 64 --max-samples 8000 --num-workers 8
```

---

### 7) Named Entity Recognition  
(datasets: `conll2003`, `wikiann_en`; models: `lstm_crf_ner`, `bilstm_crf_charcnn_ner`)

- **Preprocessing**
  - Sequence tags with padding and lengths (implemented). Optional char CNN inputs.
  - Optional: subword tokenization; casing/normalization.

- **Loss and regularization**
  - CRF negative log-likelihood (in model); or token CE if CRF disabled.
  - Dropout 0.3–0.5; word/char dropout 0.05–0.1 (extension).
  - Weight decay 1e-5–1e-4 typical.

- **Optimizers and LR**
  - Adam/SGD; scheduler ReduceLROnPlateau; clipping 1.0 (recommended).

- **Monitoring and selection**
  - Primary metric: F1.

**Recommended run commands (20 epochs, NER)**

```bash
# CoNLL-2003 / LSTM-CRF — all optimizers, plateau scheduler
python -m benchmark.main \
  --task ner --dataset conll2003 --model lstm_crf_ner \
  --optimizers all \
  --epochs 20 --batch-size 64 --max-samples 8000 --num-workers 8
```

```bash
# CoNLL-2003 / LSTM-CRF + CharCNN — all optimizers, plateau scheduler
python -m benchmark.main \
  --task ner --dataset conll2003 --model bilstm_crf_charcnn_ner \
  --optimizers all \
  --epochs 20 --batch-size 64 --max-samples 8000 --num-workers 8
```

```bash
# WikiANN (en) / LSTM-CRF — all optimizers, plateau scheduler
python -m benchmark.main \
  --task ner --dataset wikiann_en --model lstm_crf_ner \
  --optimizers all \
  --epochs 20 --batch-size 64 --max-samples 10000 --num-workers 8
```

```bash
# WikiANN (en) / LSTM-CRF + CharCNN — all optimizers, plateau scheduler
python -m benchmark.main \
  --task ner --dataset wikiann_en --model bilstm_crf_charcnn_ner \
  --optimizers all \
  --epochs 20 --batch-size 64 --max-samples 10000 --num-workers 8
```

---

### 8) Semantic Segmentation  
(datasets: `oxford_iiit_pet`, `voc2012`; models: `unet_vanilla`, `deeplabv3_resnet50`)

- **Preprocessing and augmentation**
  - Resize to 256x256, `ToTensor`, ImageNet normalization (implemented). Masks resized with nearest neighbor.
  - Optional: random scale (0.5–2.0), random crop, color jitter, blur; CutMix-style ClassMix (extension).

- **Loss and regularization**
  - CrossEntropy(`ignore=255`) + 0.5 * soft Dice (implemented).
  - Dynamic class-balanced CE weights per batch (implemented).
  - Label smoothing small (0.05) is possible but often not necessary.

- **Optimizers and LR**
  - SGD/Adam/AdamW viable; per-layer LR for backbones (extension).
  - Scheduler: cosine (default). Poly LR (extension) also common.
  - Gradient clipping: 1.0 (enabled by preset).

- **Monitoring and selection**
  - Primary metric: Dice/mIoU.

**Recommended run commands (20 epochs, segmentation)**

```bash
# Oxford-IIIT Pet / UNet — all optimizers, cosine scheduler
python -m benchmark.main \
  --task semantic_segmentation --dataset oxford_iiit_pet --model unet_vanilla \
  --optimizers all \
  --epochs 20 --batch-size 8 --max-samples 5000 --num-workers 8
```

```bash
# Oxford-IIIT Pet / DeepLabV3-ResNet50 — all optimizers, cosine scheduler
python -m benchmark.main \
  --task semantic_segmentation --dataset oxford_iiit_pet --model deeplabv3_resnet50 \
  --optimizers all \
  --epochs 20 --batch-size 8 --max-samples 5000 --num-workers 8
```

```bash
# Pascal VOC2012 / UNet — all optimizers, cosine scheduler (auto-download)
python -m benchmark.main \
  --task semantic_segmentation --dataset voc2012 --model unet_vanilla \
  --optimizers all \
  --epochs 20 --batch-size 8 --max-samples 3000 --num-workers 8
```

```bash
# Pascal VOC2012 / DeepLabV3-ResNet50 — all optimizers, cosine scheduler (auto-download)
python -m benchmark.main \
  --task semantic_segmentation --dataset voc2012 --model deeplabv3_resnet50 \
  --optimizers all \
  --epochs 20 --batch-size 8 --max-samples 3000 --num-workers 8
```

Note: Pascal VOC2012 is fetched via `torchvision.datasets.VOCSegmentation` when needed and stored under `./data/voc2012/VOCdevkit/VOC2012`. If running offline, pre-populate the dataset there and rerun the same commands.
 
Troubleshooting (VOC2012 Dice very low in early epochs)
- VOC uses 21 classes and macro Dice. Early predictions often collapse to background, so macro Dice ≈ 0 initially.
- Two quick mitigations:
  1) Use a stronger model (DeepLabV3-ResNet50):
  
  ```bash
  python -m benchmark.main \
    --task semantic_segmentation --dataset voc2012 --model deeplabv3_resnet50 \
    --optimizers all \
    --epochs 20 --batch-size 8 --max-samples 3000 --num-workers 8
  ```
  
  2) Keep UNet but switch scheduler and tweak LR for SGD-family and Adam/AdamW (helps early learning):
  
  ```bash
  python -m benchmark.main \
    --task semantic_segmentation --dataset voc2012 --model unet_vanilla \
    --optimizers all \
    --epochs 20 --batch-size 8 --max-samples 3000 --num-workers 8 \
    --scheduler plateau \
    --lr-overrides "sgd=0.01,sgd_momentum=0.01,adam=1e-3,adamw=1e-3"
  ```
 
 
---

## Per-optimizer guidance

- **SGD / SGD+momentum**
  - Strong baseline for vision; tune LR and momentum; `weight_decay=5e-4` typical.
  - Prefer cosine LR for image tasks; step decay also fine.

- **Adam / AdamW / RMSProp / Adagrad / Adadelta**
  - AdamW (`weight_decay=0.01`) is robust for Transformers; Adam is common for RNNs.
  - RMSProp is viable for some vision models; Adagrad/Adadelta for sparse/older setups.
  - In the DistilBERT sentiment commands, we provide explicit LR overrides for each optimizer to keep finetuning stable and comparable.

- **Advanced: Lion, LARS, LAMB, Yogi, AdaBelief, Adafactor, SAM/GSAM**
  - Use when matching literature for specific tasks or for stability at scale.
  - SAM/GSAM can improve generalization; require two-step updates (already supported).

---

## What’s enabled by default (quick reference)

- Best checkpointing, scheduler (auto), gradient clipping (task-based), weight decay per optimizer family.
- Label smoothing: image (0.1), sentiment (0.05), MT/Summarization (0.1).
- CIFAR-10 and MNIST augmentations; segmentation uses ImageNet normalization and CE+Dice.
- Per-run training summary text: `details.txt` saved under the run directory with preprocessing, training policy, and per-optimizer outcomes.

---

## Suggested starting presets (by family)

- **Small RNN text (e.g., LSTM sentiment)**  
  Adam/AdamW; LR around 1e-3; dropout 0.5; smoothing 0.05; `weight_decay=1e-4`; clip 1.0; plateau scheduler.

- **ResNet18 CIFAR-10**  
  SGD+momentum 0.9; LR around 0.1 with cosine; `weight_decay=5e-4`; CIFAR aug; smoothing 0.1. Consider Mixup 0.2–0.4.

- **Transformer seq2seq (IWSLT14)**  
  AdamW with LR around 5e-4; smoothing 0.1; dropout ~0.3; `weight_decay=0.01`; clip 1.0; plateau or warmup+linear.

- **Segmentation**  
  Cosine scheduler; CE(ignore) + 0.5 Dice; dynamic CE weighting; consider stronger augmentations.

---

## Notes on extending this repository

- Add additional augmentations/mixup in dataset builders or custom collate functions.
- Wire warmup schedulers where relevant (Transformers) via `main.py` scheduler factory.
- Introduce more dropout/regularization knobs in model constructors (keep defaults reasonable).
- Surfacing toggles via CLI flags helps reproducibility (follow existing patterns in `main.py`).

---
