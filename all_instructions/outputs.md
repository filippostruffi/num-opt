## Outputs and Data Layout

This document explains where datasets are stored and how benchmark outputs are organized.

### Dataset storage layout

- Root data directory: `data/` (configurable via `--data-root`, default `data`)
- Each dataset is cached/downloaded under its own subfolder:

```
data/
  mnist/
  cifar10/
  oxford_iiit_pet/
  ade20k_150/
  imdb/
  sst2/
  europarl_bilingual/
  iwslt14_en_de/
  conll2003/
  wikiann_en/
  wikitext2/
  ptb_text_only/
  cnn_dailymail/
  aeslc/
  squad_v1/
  tweet_qa/
```

Notes:
- TorchVision datasets receive their dataset-specific root under `data/<dataset_name>`.
- HuggingFace datasets use `cache_dir=data/<dataset_name>` to keep caches separated per dataset.


### Results output layout

All outputs are written under `--output-dir` (default: `./results`). For each run:

```
results/
  <epochs>/
    <task>/
      <model>/
        <dataset>/
          run_metadata.json
          <task>_<model>_<dataset>_summary.csv
          <task>_<model>_<dataset>_extensive.csv
          <optimizer_1>/
            <optimizer_1>_metrics.json
            <optimizer_1>_<primary_metric>_and_loss.png
          <optimizer_2>/
            <optimizer_2>_metrics.json
            <optimizer_2>_<primary_metric>_and_loss.png
          ...
```

Where:
- `<epochs>`: the number of training epochs for the run
- `<task>/<model>/<dataset>`: hierarchical identifier grouping a single experiment
- `<primary_metric>`: the task’s primary metric key (e.g., `accuracy`)


### Per-optimizer files

1) `<optimizer>_metrics.json`
- Validation metrics per epoch (filtered to core metrics) and timing metadata.
- Structure:
  - `history`: array of objects with keys such as `epoch`, `<primary_metric>`, `loss`, `accuracy`/`precision`/`recall`/`f1` (if applicable), `eval_runtime_s`.
  - `times`: aggregate timing information with keys like `total_runtime_s`, `epochs_to_converge`, `time_to_converge_s`, `epochs_to_plateau`, `time_to_plateau_s`.

2) `<optimizer>_<primary_metric>_and_loss.png`
- Dual y-axis plot: primary metric (left) and validation loss (right) vs. epoch.


### Run-level files (in `<task>_<model>_<dataset>` folder)

1) `run_metadata.json`
- Contains run configuration and environment info:
  - `task`, `model`, `dataset`
  - `epochs`, `batch_size`, `max_samples`, `device`
  - Learning rate policy:
    - `lr_policy`: `"per_opt"` or `"fixed"`
    - `learning_rate`: CLI value (only used if `lr_policy=fixed`)
    - `per_optimizer_lr`: map of optimizer name → effective LR used
  - `timestamp` (unix epoch seconds)

2) `<task>_<model>_<dataset>_summary.csv`
- One row per optimizer in this run. Column order:
  1. `task`
  2. `model`
  3. `dataset`
  4. `optimizer`
  5. `learning_rate`
  6. `batch_size`
  7. `primary_metric_name`
  8. `best_primary_metric_value`
  9. `best_primary_metric_epoch`
  10. `runtime_to_best_primary_metric_s`
  11. `best_val_loss_value`
  12. `best_val_loss_epoch`
  13. `runtime_to_best_val_loss_s`
  14. `runtime_to_convergence_s`
  15. `epoch_when_convergence_happened`

3) `<task>_<model>_<dataset>_extensive.csv`
- Per-epoch log across all optimizers, ordered columns:
  1. `task`
  2. `model`
  3. `dataset`
  4. `optimizer`
  5. `epoch`
  6. `primary_metric_name`
  7. `primary_metric_value`
  8. `val_loss`
  9. All other numeric validation metrics (if any; excludes per-class/macro/weighted classification metrics)
  10. `epoch_runtime_s`
  11. `eval_runtime_s`
  12. `cumulative_runtime_s`


### Legacy outputs

- Earlier comparison plots/CSVs across all optimizers have been removed to improve readability. Use the two CSVs above and per-optimizer files instead.


