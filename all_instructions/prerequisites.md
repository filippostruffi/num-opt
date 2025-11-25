## Prerequisites

- Python: 3.9+ (CPU or GPU)
- PyTorch: torch>=2.1.0 (with CUDA if available)
- TorchVision: for image/segmentation datasets and models
- Hugging Face Datasets: datasets (for data loading across NLP tasks)
- Transformers: for DistilBERT-based sentiment model
- Matplotlib: for plots
- Pandas: for tables/CSV summaries

Recommended installs:

```bash
pip install -U pip
pip install "torch>=2.1.0" torchvision datasets transformers matplotlib pandas
```

Optional (quality of life):
- certifi (macOS SSL fixes; the runner auto-configures if present)
- scikit-learn (some metrics/utilities if you expand tasks)

Data/cache prerequisites (auto-downloaded if online):
- TorchVision datasets under TORCH_HOME (default: ./data)
- Hugging Face datasets/models under HF_HOME/HF_DATASETS_CACHE (default: ./data/huggingface)


