import os
from typing import List, Dict, Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def plot_primary_and_loss(self, hist: List[Dict[str, Any]], primary_key: str, title: str, out_path: str):
        values = [x.get(primary_key, None) for x in hist]
        losses = [x.get("loss", None) for x in hist]
        epochs = list(range(1, len(values) + 1))
        plt.figure()
        ax1 = plt.gca()
        l1 = ax1.plot(epochs, values, color="tab:blue", label=primary_key)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel(primary_key, color="tab:blue")
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax2 = ax1.twinx()
        l2 = ax2.plot(epochs, losses, color="tab:red", label="val loss")
        ax2.set_ylabel("val loss", color="tab:red")
        ax2.tick_params(axis='y', labelcolor='tab:red')
        lines = l1 + l2
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="best")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, out_path))
        plt.close()

    def plot_loss(self, train_hist: List[Dict[str, Any]], val_hist: List[Dict[str, Any]], title: str, out_path: str):
        train_losses = [x.get("loss", None) for x in train_hist]
        val_losses = [x.get("loss", None) for x in val_hist]
        epochs = list(range(1, len(train_losses) + 1))
        plt.figure()
        plt.plot(epochs, train_losses, label="train loss")
        plt.plot(epochs, val_losses, label="val loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, out_path))
        plt.close()

    def plot_primary_metric(self, hist: List[Dict[str, Any]], primary_key: str, title: str, out_path: str):
        values = [x.get(primary_key, None) for x in hist]
        epochs = list(range(1, len(values) + 1))
        plt.figure()
        plt.plot(epochs, values, label=primary_key)
        plt.xlabel("Epoch")
        plt.ylabel(primary_key)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, out_path))
        plt.close()

    def plot_compare_primary(
        self,
        optimizer_to_hist: Dict[str, List[Dict[str, Any]]],
        primary_key: str,
        title: str,
        out_path: str,
    ):
        plt.figure()
        for opt, hist in optimizer_to_hist.items():
            vals = [x.get(primary_key, None) for x in hist]
            epochs = [x.get("epoch", i + 1) for i, x in enumerate(hist)]
            plt.plot(epochs, vals, label=opt)
        plt.xlabel("Epoch")
        plt.ylabel(primary_key)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, out_path))
        plt.close()

    def plot_compare_loss(
        self,
        optimizer_to_hist: Dict[str, List[Dict[str, Any]]],
        title: str,
        out_path: str,
    ):
        plt.figure()
        for opt, hist in optimizer_to_hist.items():
            vals = [x.get("loss", None) for x in hist]
            epochs = [x.get("epoch", i + 1) for i, x in enumerate(hist)]
            plt.plot(epochs, vals, label=opt)
        plt.xlabel("Epoch")
        plt.ylabel("loss")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, out_path))
        plt.close()

    def plot_multi_metrics(
        self,
        hist: List[Dict[str, Any]],
        title: str,
        out_path: str,
        exclude_keys: List[str] = None,
    ):
        if exclude_keys is None:
            exclude_keys = ["epoch", "loss", "eval_runtime_s", "epoch_runtime_s"]
        numeric_keys = set()
        for row in hist:
            for k, v in row.items():
                if k in exclude_keys:
                    continue
                if isinstance(v, (int, float)) and v is not None:
                    numeric_keys.add(k)
        if not numeric_keys:
            return
        plt.figure()
        epochs = [x.get("epoch", i + 1) for i, x in enumerate(hist)]
        for k in sorted(numeric_keys):
            vals = [x.get(k, None) for x in hist]
            plt.plot(epochs, vals, label=k)
        # also include loss
        if "loss" in {kk for row in hist for kk in row.keys()}:
            vals = [x.get("loss", None) for x in hist]
            plt.plot(epochs, vals, label="loss")
        plt.xlabel("Epoch")
        plt.ylabel("value")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, out_path))
        plt.close()


