from typing import Any
import torch
from torch import nn
from benchmark.config.registry import register_model


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(64, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.out = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        xb = self.bottleneck(self.pool2(x2))
        x = self.up2(xb)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))
        return self.out(x)


@register_model("unet_vanilla")
def build_model(task: Any, dataset: Any) -> nn.Module:
    # Infer input channels and number of classes robustly.
    # Prefer dataset-provided num_classes when available.
    x0, y0 = dataset[0]
    in_ch = x0.shape[0]
    num_classes = None
    base_ds = dataset
    # Unwrap torch.utils.data.Subset-style wrappers to access underlying dataset attributes
    try:
        depth_guard = 0
        while hasattr(base_ds, "dataset") and depth_guard < 5:
            base_ds = getattr(base_ds, "dataset")
            depth_guard += 1
    except Exception:
        base_ds = dataset
    if hasattr(base_ds, "num_classes"):
        try:
            nc = int(getattr(base_ds, "num_classes"))
            if nc > 1:
                num_classes = nc
        except Exception:
            num_classes = None
    if num_classes is None:
        max_label = -1
        # Probe up to first 64 samples (or dataset length if smaller) to reduce underestimation risk
        probe_count = min(64, len(dataset))
        for i in range(probe_count):
            _, yi = dataset[i]
            if hasattr(yi, "to"):
                yi_flat = yi.view(-1)
                yi_valid = yi_flat[yi_flat != 255]
                if yi_valid.numel() > 0:
                    max_label = max(max_label, int(yi_valid.max().item()))
            else:
                try:
                    import torch
                    yi_t = torch.as_tensor(yi)
                    yi_flat = yi_t.view(-1)
                    yi_valid = yi_flat[yi_flat != 255]
                    if yi_valid.numel() > 0:
                        max_label = max(max_label, int(yi_valid.max().item()))
                except Exception:
                    continue
        import torch
        num_classes = (max_label + 1) if max_label >= 0 else (int(torch.max(y0).item()) + 1)
    return UNet(in_channels=in_ch, num_classes=num_classes)


