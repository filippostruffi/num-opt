## Numerical Optimizer Benchmark (Multi-Task)

This repository benchmarks optimization algorithms across vision and NLP tasks with runnable baselines.

Start here:
- Prerequisites: [all_instructions/prerequisites.md](all_instructions/prerequisites.md)
- How to run: [all_instructions/how_to_run.md](all_instructions/how_to_run.md)
- Tasks overview: [all_instructions/tasks.md](all_instructions/tasks.md)
- Datasets (by task): [all_instructions/datasets.md](all_instructions/datasets.md)
- Models (by task): [all_instructions/models.md](all_instructions/models.md)
- Optimizers: [all_instructions/optimizers.md](all_instructions/optimizers.md)
- Metrics: [all_instructions/metrics.md](all_instructions/metrics.md)
- Commands index (all combinations): [all_instructions/commands.md](all_instructions/commands.md)
- Run everything (script): [all_instructions/run_all.sh](all_instructions/run_all.sh)

Quick start:
```bash
python -m benchmark.main --task <task> --dataset <dataset> --model <model> --optimizers all --epochs 1 --batch-size 64 --max-samples 2 --output-dir ./results
```

