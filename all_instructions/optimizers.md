## Optimizers

Library (torch.optim):
- SGD / SGD with momentum: vanilla and momentum variants.
- RMSprop: EMA-normalized updates for non-stationary targets.
- Adagrad: Per-parameter learning-rate scaling via accumulated squared grads.
- Adadelta: Adagrad variant with decaying average of squared updates.
- Adam: Adaptive moments (m, v) with bias correction.
- AdamW: Adam with decoupled weight decay.
- RAdam: Adam with rectified variance.

Custom (implemented in-repo; file: `benchmark/optimizers/advanced.py`)
- Lion
  - Paper: Chen et al., 2023 (Google Research)
  - Idea: sign-descent on EMA of gradients.
  - Update (conceptual): m_t = β₁ m_{t−1} + (1−β₁) g_t; θ ← θ − lr · sign(m_t)
  - Implementation: sign(m) step with EMA; decoupled WD.
  - Notes: matches common Lion update; minimal variant.
- LARS
  - Paper: You et al., 2017
  - Idea: layer-wise trust ratio η·||w||/||g|| with momentum.
  - Update: v ← μ v + trust_ratio · g; θ ← θ − lr · v, with trust_ratio = η · ||w|| / ||g||
  - Implementation: trust ratio with 1D param exclusion; momentum; optional WD.
  - Notes: faithful minimal.
- LAMB
  - Paper: You et al., 2019
  - Idea: Adam step scaled by ||w||/||u|| layer-wise.
  - Update: u = Adam(m̂, v̂); trust_ratio = ||w||/||u||; θ ← θ − lr · trust_ratio · u
  - Implementation: Adam with bias corrections; decoupled WD; 1D exclusion for trust ratio.
  - Notes: faithful core; no extra clipping/windowing from some refs.
- AdaBelief
  - Paper: Zhuang et al., 2020
  - Idea: v tracks (g − m)^2; update m/√v.
  - Update: m_t = β₁ m_{t−1} + (1−β₁) g_t; s_t = β₂ s_{t−1} + (1−β₂)(g_t − m_t)²; θ ← θ − lr · m̂ / (√ŝ + ε)
  - Implementation: bias corrections; decoupled WD.
  - Notes: faithful core; minimal extras.
- Yogi
  - Paper: Zaheer et al., 2018
  - Idea: v_t = v_{t−1} − (1−β2)·sign(v_{t−1} − g^2)·g^2; stabilized second moment.

### Default learning rates used by CLI (per-optimizer)
These defaults are applied automatically when you omit `--learning-rate` (the CLI defaults to a per-optimizer LR policy):

- sgd: 0.05
- sgd_momentum: 0.01
- rmsprop: 0.001
- adagrad: 0.01
- adadelta: 1.0
- adam: 0.001
- adamw: 0.001
- radam: 0.001
- lion: 0.0001
- lars: 0.1
- lamb: 0.001
- adabelief: 0.001
- yogi: 0.001
- adafactor: 0.001
- sam: 0.001
- gsam: 0.001

Notes:
- You can force a single fixed LR for all optimizers using `--lr-policy fixed --learning-rate <value>`.
- You can override specific optimizers while keeping per-optimizer defaults via `--lr-overrides "sgd=0.05,lars=0.1"`.
  - Update: m_t = β₁ m_{t−1} + (1−β₁) g_t; v_t = v_{t−1} − (1−β₂) sign(v_{t−1}−g²) ⊙ g²; θ ← θ − lr · m̂/(√v̂+ε)
  - Implementation: exact variance rule with bias corrections; decoupled WD.
  - Notes: faithful.
- Adafactor
  - Paper: Shazeer & Stern, 2018
  - Idea: factored second-moment estimates; relative step.
  - Update: v ≈ r ⊗ c (factored second moment), ĝ = g / √(v+ε₂small); clip by RMS; relative step lr ∝ 1/√t (optional)
  - Implementation: factored v for matrices; RMS clipping; optional relative_step/warmup; non-factored for vectors.
  - Notes: close but simplified schedules vs full paper.
- SAM
  - Paper: Foret et al., 2021
  - Idea: perturb in gradient direction (radius ρ), then optimize sharpness-aware objective.
  - Implementation: two-step closure; AdamW base by default.
  - Notes: faithful minimal.
- GSAM
  - Paper: Zhuang et al., 2022
  - Idea: gradient-guided SAM with alignment term.
  - Implementation: approximated via scaling ρ by (1+γ); not full GSAM alignment.
  - Notes: approximation, not full method.

References:
- LARS: https://arxiv.org/abs/1708.03888
- LAMB: https://arxiv.org/abs/1904.00962
- AdaBelief: https://arxiv.org/abs/2010.07468
- Yogi: https://papers.nips.cc/paper_files/paper/2018/hash/90365351ccc7437a1309dc64e4db32a3-Abstract.html
- Adafactor: https://arxiv.org/abs/1804.04235
- SAM: https://arxiv.org/abs/2010.01412
- GSAM: https://arxiv.org/abs/2205.14083
- Lion: https://arxiv.org/abs/2302.06675


