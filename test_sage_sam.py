"""SAGE-SAM test-set evaluation entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from Model.sage_model import DualSegmentor
from dataloader.builders import SAGEImageDataset, resolve_dataset_root, validate_dataset_layout
from engine.checkpoint import _safe_load
from engine.evaluator import _fused_logits
from utils.utils import multiclass_segmentation_metrics, safe_nanmean


class TerminalProgress:
    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self.last_len = 0

    def update(self, message: str) -> None:
        padding = max(self.last_len - len(message), 0)
        self.stream.write("\r" + message + (" " * padding))
        self.stream.flush()
        self.last_len = len(message)

    def finish(self, message: str = "") -> None:
        if self.last_len:
            self.stream.write("\r" + (" " * self.last_len) + "\r")
            self.last_len = 0
        if message:
            self.stream.write(message + "\n")
        self.stream.flush()


def add_arg(parser, *names, **kwargs):
    parser.add_argument(*names, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a SAGE-SAM checkpoint on val/test split.")
    add_arg(parser, "--data_path", "--data-path", dest="data_path", required=True)
    add_arg(parser, "--dataset", required=True)
    add_arg(parser, "--checkpoint", required=True)
    add_arg(parser, "--split", choices=["val", "test"], default="test")
    add_arg(parser, "--output_dir", "--output-dir", dest="output_dir", default="./outputs")
    add_arg(parser, "--experiment_name", "--experiment-name", dest="experiment_name", default="SAGE_SAM_3Class")
    add_arg(parser, "--metrics_file", "--metrics-file", dest="metrics_file")
    add_arg(parser, "--num_classes", "--num-classes", dest="num_classes", type=int)
    add_arg(parser, "--in_channels", "--in-channels", dest="in_channels", type=int)
    add_arg(parser, "--image_size", "--image-size", dest="image_size", type=int)
    add_arg(parser, "--num_workers", "--num-workers", dest="num_workers", type=int, default=0)
    add_arg(parser, "--device", default=None)
    return parser


def progress_bar(completed: int, total: int, width: int = 28) -> str:
    total = max(total, 1)
    filled = min(width, max(0, round(width * completed / total)))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def format_seconds(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    minutes, seconds = divmod(seconds, 60)
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def choose_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def checkpoint_args(payload: dict) -> dict:
    args = payload.get("args") or {}
    return args if isinstance(args, dict) else {}


def int_arg(cli_value, ckpt_args: dict, key: str, default: int) -> int:
    if cli_value is not None:
        return int(cli_value)
    if key in ckpt_args and ckpt_args[key] is not None:
        return int(ckpt_args[key])
    return default


def load_model(checkpoint: str | Path, device: torch.device, num_classes: int, in_channels: int) -> tuple[DualSegmentor, dict]:
    payload = _safe_load(checkpoint, map_location="cpu")
    model = DualSegmentor(in_channels=in_channels, num_classes=num_classes).to(device)
    state = payload.get("model", payload)
    current = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state.items():
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            filtered[key] = value
        elif key in current:
            skipped.append(key)
    report = model.load_state_dict(filtered, strict=False)
    if skipped:
        raise ValueError(f"Checkpoint shape mismatch for keys: {skipped[:10]}")
    model.eval()
    return model, {
        "loaded_keys": len(filtered),
        "missing_keys": list(report.missing_keys),
        "unexpected_keys": list(report.unexpected_keys),
    }


@torch.no_grad()
def evaluate_split(model: DualSegmentor, loader: DataLoader, device: torch.device, num_classes: int) -> dict:
    progress = TerminalProgress()
    start = time.time()
    rows = {"unet": [], "vnet": [], "fused": []}
    total = len(loader)
    for idx, batch in enumerate(loader, start=1):
        image = batch["image"].to(device)
        label = batch["mask"]
        out = model(image, return_features=False)
        logits_a = out["logits_a"]
        logits_b = out["logits_b"]
        fused = _fused_logits(logits_a, logits_b)
        preds = {
            "unet": torch.argmax(logits_a, dim=1),
            "vnet": torch.argmax(logits_b, dim=1),
            "fused": torch.argmax(fused, dim=1),
        }
        for key, pred in preds.items():
            rows[key].append(
                multiclass_segmentation_metrics(
                    label.squeeze(0).detach().cpu().numpy(),
                    pred.squeeze(0).detach().cpu().numpy(),
                    num_classes=num_classes,
                )
            )
        elapsed = time.time() - start
        eta = (elapsed / max(idx, 1)) * max(total - idx, 0)
        current_dice = rows["fused"][-1]["avg_dice"]
        progress.update(
            f"{progress_bar(idx, total)} {idx}/{total} "
            f"fused_dice={current_dice:.4f} "
            f"avg_fused_dice={safe_nanmean([r['avg_dice'] for r in rows['fused']]):.4f} "
            f"eta={format_seconds(eta)}"
        )
    progress.finish()
    result = {
        "sample_count": total,
        "unet_avg_dice": safe_nanmean([r["avg_dice"] for r in rows["unet"]]),
        "vnet_avg_dice": safe_nanmean([r["avg_dice"] for r in rows["vnet"]]),
        "fused_avg_dice": safe_nanmean([r["avg_dice"] for r in rows["fused"]]),
        "fused_avg_iou": safe_nanmean([r["avg_iou"] for r in rows["fused"]]),
        "fused_avg_hd95": safe_nanmean([r["avg_hd95"] for r in rows["fused"]]),
    }
    for cls in range(1, num_classes):
        result[f"class_{cls}_dice"] = safe_nanmean([r[f"class_{cls}_dice"] for r in rows["fused"]])
        result[f"class_{cls}_iou"] = safe_nanmean([r[f"class_{cls}_iou"] for r in rows["fused"]])
        result[f"class_{cls}_hd95"] = safe_nanmean([r[f"class_{cls}_hd95"] for r in rows["fused"]])
    return result


def main() -> None:
    args = build_parser().parse_args()
    device = choose_device(args.device)
    payload = _safe_load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint_args(payload)
    num_classes = int_arg(args.num_classes, ckpt_args, "num_classes", 3)
    in_channels = int_arg(args.in_channels, ckpt_args, "in_channels", 3)
    image_size = int_arg(args.image_size, ckpt_args, "image_size", 256)
    dataset_root = resolve_dataset_root(args.data_path, args.dataset)
    validate_dataset_layout(dataset_root)
    model, load_report = load_model(args.checkpoint, device, num_classes, in_channels)
    dataset = SAGEImageDataset(dataset_root, args.split, num_classes, image_size, has_mask=True, in_channels=in_channels)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=args.num_workers)
    print(f"Testing split={args.split} samples={len(dataset)} checkpoint={args.checkpoint}")
    metrics = evaluate_split(model, loader, device, num_classes)
    output = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dataset_root": str(dataset_root),
        "split": args.split,
        "num_classes": num_classes,
        "in_channels": in_channels,
        "image_size": image_size,
        "load_report": load_report,
        "metrics": metrics,
    }
    metrics_file = Path(args.metrics_file) if args.metrics_file else Path(args.output_dir) / args.experiment_name / "metrics" / f"{args.split}_metrics.json"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Saved metrics: {metrics_file}")


if __name__ == "__main__":
    main()
