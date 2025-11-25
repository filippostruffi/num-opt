## How to Run

1) Create and activate a virtual environment (optional but recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies
```bash
pip install -U pip
pip install "torch>=2.1.0" torchvision datasets transformers matplotlib pandas
```

3) (Optional) Configure local caches
```bash
mkdir -p ./data ./data/huggingface ./data/torchtext
export TORCH_HOME=./data
export HF_HOME=./data/huggingface
export HF_DATASETS_CACHE=./data/huggingface
export TOKENIZERS_PARALLELISM=false
```

4) Run from repository root (module mode preferred)
```bash
python -m benchmark.main \
  --task <task> \
  --dataset <dataset> \
  --model <model> \
  --optimizers <comma_separated_optimizers|all> \
  --epochs 1 \
  --batch-size 64 \
  --output-dir ./results \
  --max-samples 2
```

Notes:
- Per-optimizer default learning rates are used by default (no need to set `--learning-rate`).
- Use `--optimizers all` to run every registered optimizer on the same setup.
- Results: metrics JSON/CSVs and plots under `--output-dir/<epochs>/<task>/<model>/<dataset>/`.
- If downloads fail (network restricted), pre-populate `./data` and set the env vars above.

5) Run all combinations (script)
```bash
# Option A: run via bash
bash all_instructions/run_all.sh

# Option B: make executable once, then run
chmod +x all_instructions/run_all.sh
./all_instructions/run_all.sh

# Optional overrides (defaults: BS=64, EPOCHS=1, MAX_SAMPLES=2)
BS=32 EPOCHS=1 MAX_SAMPLES=2 bash all_instructions/run_all.sh
```

Advanced (optional):
- Force a fixed LR for all optimizers: add `--lr-policy fixed --learning-rate <value>` to the command.
- Override specific optimizers in per-opt policy: `--lr-overrides "sgd=0.05,lars=0.1"`.


