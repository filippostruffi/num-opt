import argparse
import os
import json
import csv
import time
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader
import ssl

from benchmark.config.registry import (
    TASK_REGISTRY,
    DATASET_REGISTRY,
    MODEL_REGISTRY,
    build_optimizers_from_names,
)
from benchmark.reporting.logger import ResultLogger
from benchmark.reporting.plots import Plotter
from benchmark.reporting.tables import TableWriter
from benchmark.training.trainer import Trainer
from benchmark.optimizers.factory import OPTIMIZER_REGISTRY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Task Optimizer Benchmark")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., image_classification)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., cifar10)")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., cnn_scratch)")
    parser.add_argument("--optimizers", type=str, required=True, help="Comma-separated optimizer names")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Used only if --lr-policy fixed")
    parser.add_argument(
        "--lr-policy",
        type=str,
        default="per_opt",
        choices=["fixed", "per_opt"],
        help="Learning rate selection policy. 'per_opt' uses defaults per optimizer; 'fixed' uses --learning-rate for all.",
    )
    parser.add_argument(
        "--lr-overrides",
        type=str,
        default=None,
        help="Comma-separated overrides for per-opt policy, e.g. 'sgd=0.05,lars=0.1'. Names are case-insensitive.",
    )
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples per split")
    parser.add_argument("--data-root", type=str, default="data", help="Root directory for datasets")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plateau-patience", type=int, default=3, help="Plateau patience (epochs without improvement)")
    return parser.parse_args()


def configure_ssl_certificates() -> None:
    # Ensure urllib/ssl and requests use a valid CA bundle (fixes macOS cert issues)
    try:
        import certifi  # type: ignore
        ca_path = certifi.where()
        if not os.environ.get("SSL_CERT_FILE"):
            os.environ["SSL_CERT_FILE"] = ca_path
        if not os.environ.get("REQUESTS_CA_BUNDLE"):
            os.environ["REQUESTS_CA_BUNDLE"] = ca_path
        try:
            context = ssl.create_default_context(cafile=ca_path)
            ssl._create_default_https_context = lambda: context  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        # certifi not available; rely on system defaults
        pass


def main() -> None:
    args = parse_args()
    configure_ssl_certificates()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    # Ensure all datasets cache/end under --data-root
    try:
        os.makedirs(args.data_root, exist_ok=True)
        # TorchVision/Torch defaults
        if not os.environ.get("TORCH_HOME"):
            os.environ["TORCH_HOME"] = args.data_root
        # TorchText (if used anywhere)
        if not os.environ.get("TORCHTEXT_HOME"):
            os.environ["TORCHTEXT_HOME"] = os.path.join(args.data_root, "torchtext")
        # HuggingFace caches (datasets + models) under --data-root/huggingface
        hf_cache = os.path.join(args.data_root, "huggingface")
        os.makedirs(hf_cache, exist_ok=True)
        if not os.environ.get("HF_DATASETS_CACHE"):
            os.environ["HF_DATASETS_CACHE"] = hf_cache
        # Prefer HF_HOME (recommended by HF) over deprecated TRANSFORMERS_CACHE
        if not os.environ.get("HF_HOME"):
            os.environ["HF_HOME"] = hf_cache
        # Silence tokenizers parallelism fork warnings in dataloader workers
        if not os.environ.get("TOKENIZERS_PARALLELISM"):
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
    except Exception:
        pass

    # Ensure all registered via side-effect imports
    from benchmark.config.registry import ensure_imports
    ensure_imports()

    if args.task not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {args.task}. Available: {list(TASK_REGISTRY.keys())}")
    if args.dataset not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {args.dataset}. Available: {list(DATASET_REGISTRY.keys())}")
    if args.model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {args.model}. Available: {list(MODEL_REGISTRY.keys())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_cls = TASK_REGISTRY[args.task]
    task = task_cls()

    # Build datasets
    train_ds, val_ds = DATASET_REGISTRY[args.dataset](max_samples=args.max_samples, data_root=args.data_root)
    collate_fn = getattr(task, "collate_fn", None)

    drop_last = True if args.task == "semantic_segmentation" else False
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn
    )

    # Build model
    model = MODEL_REGISTRY[args.model](task=task, dataset=train_ds).to(device)
    # If segmentation, align metrics num_classes to dataset/model
    if args.task == "semantic_segmentation":
        try:
            # Prefer inferring from dataset masks to match current run's labels
            _, m0 = train_ds[0]
            if torch.is_tensor(m0):
                mvalid = m0[m0 != 255]
                num_classes = int(mvalid.max().item()) + 1 if mvalid.numel() > 0 else int(m0.max().item()) + 1
                from benchmark.metrics.segmentation_metrics import SegmentationMetrics
                task.metric = SegmentationMetrics(num_classes=num_classes, ignore_index=255)
        except Exception:
            try:
                out_ch = None
                if hasattr(model, "out") and hasattr(model.out, "out_channels"):
                    out_ch = model.out.out_channels
                elif hasattr(model, "classifier") and hasattr(model.classifier, "out_channels"):
                    out_ch = model.classifier.out_channels
                if out_ch is not None:
                    from benchmark.metrics.segmentation_metrics import SegmentationMetrics
                    task.metric = SegmentationMetrics(num_classes=out_ch, ignore_index=255)
            except Exception:
                pass

    # Prepare result logging
    run_metadata: Dict[str, Any] = {
        "task": args.task,
        "dataset": args.dataset,
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lr_policy": args.lr_policy,
        "max_samples": args.max_samples,
        "device": str(device),
        "timestamp": int(time.time()),
    }
    logger = ResultLogger(args.output_dir, run_metadata)
    # Organized output root: results/<epochs>/<task>/<model>/<dataset>
    run_root_dir = os.path.join(
        args.output_dir,
        str(args.epochs),
        args.task,
        args.model,
        args.dataset,
    )
    os.makedirs(run_root_dir, exist_ok=True)

    # Build optimizers list
    optimizer_names: List[str] = [name.strip() for name in args.optimizers.split(",") if name.strip()]
    if len(optimizer_names) == 1 and optimizer_names[0].lower() in ("all", "*"):
        optimizer_names = list(OPTIMIZER_REGISTRY.keys())
        used_all_flag = True
    else:
        used_all_flag = False

    # Per-optimizer default learning rates
    DEFAULT_LRS: Dict[str, float] = {
        # SGD family
        "sgd": 5e-2,
        "sgd_momentum": 1e-2,
        # Adaptive (torch)
        "rmsprop": 1e-3,
        "adagrad": 1e-2,
        "adadelta": 1.0,
        "adam": 1e-3,
        "adamw": 1e-3,
        "radam": 1e-3,
        # Advanced/custom
        "lion": 1e-4,
        "lars": 1e-1,
        "lamb": 1e-3,
        "adabelief": 1e-3,
        "yogi": 1e-3,
        "adafactor": 1e-3,
        # Sharpness-aware
        "sam": 1e-3,
        "gsam": 1e-3,
    }

    def parse_lr_overrides(spec: str) -> Dict[str, float]:
        mapping: Dict[str, float] = {}
        if not spec:
            return mapping
        for item in spec.split(","):
            if not item.strip():
                continue
            if "=" not in item:
                continue
            k, v = item.split("=", 1)
            k = k.strip().lower()
            try:
                mapping[k] = float(v.strip())
            except Exception:
                continue
        return mapping

    # Build lr map according to policy
    lr_map: Dict[str, float]
    if args.lr_policy == "fixed":
        lr_map = {opt: float(args.learning_rate) for opt in optimizer_names}
    else:
        lr_map = {opt: DEFAULT_LRS.get(opt, float(args.learning_rate)) for opt in optimizer_names}
        if args.lr_overrides:
            overrides = parse_lr_overrides(args.lr_overrides)
            for k, v in overrides.items():
                if k in lr_map:
                    lr_map[k] = v
    # Record chosen per-optimizer lrs for traceability
    run_metadata["per_optimizer_lr"] = {opt: lr_map.get(opt) for opt in optimizer_names}
    logger = ResultLogger(args.output_dir, run_metadata)

    # Summary and extensive CSV rows
    summary_rows: List[Dict[str, Any]] = []
    extensive_rows: List[Dict[str, Any]] = []

    # If CSVs from a prior run exist for this run_root_dir, load them so we can
    # update incrementally after each optimizer finishes.
    base_name = f"{args.task}_{args.model}_{args.dataset}"
    summary_csv_path = os.path.join(run_root_dir, f"{base_name}_summary.csv")
    extensive_csv_path = os.path.join(run_root_dir, f"{base_name}_extensive.csv")
    if os.path.exists(summary_csv_path):
        try:
            with open(summary_csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    summary_rows.append(row)
        except Exception:
            pass
    if os.path.exists(extensive_csv_path):
        try:
            with open(extensive_csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    extensive_rows.append(row)
        except Exception:
            pass

    # Keep per-optimizer histories no longer needed globally (plots simplified)

    for opt_name in optimizer_names:
        # If this is an "all" run and results for this optimizer already exist, skip it
        opt_dir = os.path.join(run_root_dir, opt_name)
        if used_all_flag and os.path.isdir(opt_dir):
            print(f"Skipping optimizer '{opt_name}' (existing results detected at {opt_dir})")
            continue
        # Fresh model for each optimizer
        model.apply(task.reset_parameters)
        optimizer = build_optimizers_from_names(model.parameters(), [opt_name], lr=lr_map)[0]

        trainer = Trainer(task=task, device=device, logger=logger)
        train_hist, val_hist, times = trainer.fit(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            optimizer_name=opt_name,
            plateau_patience=args.plateau_patience,
        )

        # Prepare per-optimizer directory and plot (primary metric + val loss)
        os.makedirs(opt_dir, exist_ok=True)
        opt_plotter = Plotter(opt_dir)
        opt_plotter.plot_primary_and_loss(
            val_hist,
            task.primary_metric_key(),
            title=f"{args.task}/{args.model}/{args.dataset} [{opt_name}]",
            out_path=f"{opt_name}_{task.primary_metric_key()}_and_loss.png",
        )

        # Save per-optimizer JSON with original (non per-class) metrics only
        def _filter_metrics_row(row: Dict[str, Any]) -> Dict[str, Any]:
            # Keep core keys; drop per-class and aggregated extras introduced for classification
            keep_keys = {"epoch", "loss", "eval_runtime_s", "accuracy", "precision", "recall", "f1"}
            filtered = {}
            for k, v in row.items():
                if k in keep_keys:
                    filtered[k] = v
                # Always retain the primary metric key even if not in keep_keys
                elif k == task.primary_metric_key():
                    filtered[k] = v
                # Drop per-class and aggregated keys
                elif k.startswith("precision_c") or k.startswith("recall_c") or k.startswith("f1_c") or k.startswith("support_c"):
                    continue
                elif k.startswith("macro_") or k.startswith("weighted_"):
                    continue
            return filtered

        val_hist_filtered = [{**_filter_metrics_row(r)} for r in val_hist]
        # Save under optimizer-specific directory
        with open(os.path.join(opt_dir, f"{opt_name}_metrics.json"), "w") as f:
            json.dump({"history": val_hist_filtered, "times": times}, f, indent=2)

        # Aggregate for comparison (final + best across epochs)
        primary_key = task.primary_metric_key()
        best_primary_val = None
        best_primary_epoch = None
        best_loss_val = None
        best_loss_epoch = None
        # Cumulative runtime per epoch (train + eval)
        cumulative_runtime: List[float] = []
        cumu = 0.0
        for i, vr in enumerate(val_hist, start=1):
            pv = vr.get(primary_key, None)
            if pv is not None:
                if (best_primary_val is None) or task.is_better(pv, best_primary_val):
                    best_primary_val = pv
                    best_primary_epoch = i
            lv = vr.get("loss", None)
            if lv is not None:
                if (best_loss_val is None) or (lv < best_loss_val):
                    best_loss_val = lv
                    best_loss_epoch = i
            tr_et = train_hist[i - 1].get("epoch_runtime_s", 0.0)
            ev_et = val_hist[i - 1].get("eval_runtime_s", 0.0)
            cumu += float(tr_et) + float(ev_et)
            cumulative_runtime.append(cumu)

        # Build summary row for this optimizer
        runtime_to_best_primary = cumulative_runtime[best_primary_epoch - 1] if best_primary_epoch is not None else None
        runtime_to_best_loss = cumulative_runtime[best_loss_epoch - 1] if best_loss_epoch is not None else None
        # Values at convergence epoch (if any)
        conv_epoch = times.get("epochs_to_converge")
        primary_at_convergence = None
        val_loss_at_convergence = None
        if isinstance(conv_epoch, int) and 1 <= conv_epoch <= len(val_hist):
            conv_row = val_hist[conv_epoch - 1]
            primary_at_convergence = conv_row.get(primary_key, None)
            val_loss_at_convergence = conv_row.get("loss", None)
        summary_rows.append(
            {
                "task": args.task,
                "model": args.model,
                "dataset": args.dataset,
                "optimizer": opt_name,
                "learning_rate": lr_map.get(opt_name),
                "batch_size": args.batch_size,
                "primary_metric_name": primary_key,
                "best_primary_metric_value": best_primary_val,
                "best_primary_metric_epoch": best_primary_epoch,
                "runtime_to_best_primary_metric_s": runtime_to_best_primary,
                "best_val_loss_value": best_loss_val,
                "best_val_loss_epoch": best_loss_epoch,
                "runtime_to_best_val_loss_s": runtime_to_best_loss,
                "runtime_to_convergence_s": times.get("time_to_converge_s"),
                "epoch_when_convergence_happened": times.get("epochs_to_converge"),
                "primary_metric_at_convergence": primary_at_convergence,
                "val_loss_at_convergence": val_loss_at_convergence,
            }
        )

        # Extensive per-epoch rows (validation metrics only; exclude per-class/macro/weighted)
        for i, vr in enumerate(val_hist, start=1):
            row: Dict[str, Any] = {
                "task": args.task,
                "model": args.model,
                "dataset": args.dataset,
                "optimizer": opt_name,
                "epoch": i,
                "primary_metric_name": primary_key,
                "primary_metric_value": vr.get(primary_key, None),
                "val_loss": vr.get("loss", None),
                "epoch_runtime_s": train_hist[i - 1].get("epoch_runtime_s", None),
                "eval_runtime_s": vr.get("eval_runtime_s", None),
                "cumulative_runtime_s": cumulative_runtime[i - 1] if i - 1 < len(cumulative_runtime) else None,
            }
            exclude_keys = {"epoch", "loss", "eval_runtime_s", primary_key}
            for k, v in vr.items():
                if k in exclude_keys:
                    continue
                if isinstance(v, (int, float)) and v is not None:
                    if k.startswith("precision_c") or k.startswith("recall_c") or k.startswith("f1_c") or k.startswith("support_c"):
                        continue
                    if k.startswith("macro_") or k.startswith("weighted_"):
                        continue
                    row[k] = v
            extensive_rows.append(row)

        # Write/update CSVs after this optimizer finishes, keeping naming and path the same
        summary_order = [
            "task",
            "model",
            "dataset",
            "optimizer",
            "learning_rate",
            "batch_size",
            "primary_metric_name",
            "best_primary_metric_value",
            "best_primary_metric_epoch",
            "runtime_to_best_primary_metric_s",
            "best_val_loss_value",
            "best_val_loss_epoch",
            "runtime_to_best_val_loss_s",
            "runtime_to_convergence_s",
            "epoch_when_convergence_happened",
            "primary_metric_at_convergence",
            "val_loss_at_convergence",
        ]
        extensive_fixed = [
            "task",
            "model",
            "dataset",
            "optimizer",
            "epoch",
            "primary_metric_name",
            "primary_metric_value",
            "val_loss",
        ]
        runtime_fields = ["epoch_runtime_s", "eval_runtime_s", "cumulative_runtime_s"]
        other_metric_keys = []
        if extensive_rows:
            known = set(extensive_fixed + runtime_fields)
            collect = set()
            for r in extensive_rows:
                for k in r.keys():
                    if k not in known:
                        collect.add(k)
            other_metric_keys = sorted(list(collect))
        extensive_order = extensive_fixed + other_metric_keys + runtime_fields
        tw = TableWriter(run_root_dir)
        tw.save_table_with_order(summary_rows, f"{base_name}_summary.csv", summary_order)
        tw.save_table_with_order(extensive_rows, f"{base_name}_extensive.csv", extensive_order)

    # Save the two overall CSVs under run root dir with explicit column order
    base_name = f"{args.task}_{args.model}_{args.dataset}"
    summary_order = [
        "task",
        "model",
        "dataset",
        "optimizer",
        "learning_rate",
        "batch_size",
        "primary_metric_name",
        "best_primary_metric_value",
        "best_primary_metric_epoch",
        "runtime_to_best_primary_metric_s",
        "best_val_loss_value",
        "best_val_loss_epoch",
        "runtime_to_best_val_loss_s",
        "runtime_to_convergence_s",
        "epoch_when_convergence_happened",
        "primary_metric_at_convergence",
        "val_loss_at_convergence",
    ]
    # For extensive, start with fixed fields, then other metric keys in stable order, then runtime fields
    extensive_fixed = [
        "task",
        "model",
        "dataset",
        "optimizer",
        "epoch",
        "primary_metric_name",
        "primary_metric_value",
        "val_loss",
    ]
    runtime_fields = ["epoch_runtime_s", "eval_runtime_s", "cumulative_runtime_s"]
    # Collect other metric keys from rows
    other_metric_keys = []
    if extensive_rows:
        known = set(extensive_fixed + runtime_fields)
        collect = set()
        for r in extensive_rows:
            for k in r.keys():
                if k not in known:
                    collect.add(k)
        other_metric_keys = sorted(list(collect))
    extensive_order = extensive_fixed + other_metric_keys + runtime_fields
    tw = TableWriter(run_root_dir)
    tw.save_table_with_order(summary_rows, f"{base_name}_summary.csv", summary_order)
    tw.save_table_with_order(extensive_rows, f"{base_name}_extensive.csv", extensive_order)

    # Save run metadata
    with open(os.path.join(run_root_dir, "run_metadata.json"), "w") as f:
        json.dump(run_metadata, f, indent=2)

    print(f"Done. Outputs saved to: {run_root_dir}")


if __name__ == "__main__":
    main()


