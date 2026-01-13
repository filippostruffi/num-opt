## Numerical Optimizer Benchmark (Multi-Task)

This repository benchmarks optimization algorithms across vision and NLP tasks with runnable baselines.

Start here:
- How to run: [all_instructions/how_to_run.md](all_instructions/how_to_run.md)
- Optimizers: [all_instructions/optimizers.md](all_instructions/optimizers.md)

Quick start:
```bash
python -m benchmark.main --task <task> --dataset <dataset> --model <model> --optimizers all --epochs 1 --batch-size 64 --max-samples 2 --output-dir ./results
```

