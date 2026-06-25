"""Evaluation helpers. Inference never loads SAM or the structure cache."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from Model.sage_model import DualSegmentor
from utils.utils import multiclass_segmentation_metrics, safe_nanmean


@torch.no_grad()
def predict_logits(model: DualSegmentor, image: torch.Tensor, branch_fusion: str = "mean") -> torch.Tensor:
    model.eval()
    out = model(image, return_features=False)
    logits_a = out["logits_a"]
    logits_b = out["logits_b"]
    if logits_a is None or logits_b is None:
        raise RuntimeError("DualSegmentor inference requires both branches")
    if branch_fusion == "a":
        return logits_a
    if branch_fusion == "b":
        return logits_b
    return 0.5 * (logits_a + logits_b)


def inference_summary() -> dict:
    return {
        "inference_uses_sam": False,
        "inference_reads_precomputed_structure": False,
        "active_model": "DualSegmentor",
    }


def _fused_logits(logits_a: torch.Tensor, logits_b: torch.Tensor, reliability_a=None, reliability_b=None) -> torch.Tensor:
    if reliability_a is None or reliability_b is None:
        return 0.5 * (logits_a + logits_b)
    ra = torch.tensor(reliability_a, device=logits_a.device, dtype=logits_a.dtype).view(1, -1, 1, 1)
    rb = torch.tensor(reliability_b, device=logits_b.device, dtype=logits_b.dtype).view(1, -1, 1, 1)
    denom = (ra + rb).clamp_min(1e-6)
    return (logits_a * ra + logits_b * rb) / denom


@torch.no_grad()
def evaluate_multiclass(
    model: DualSegmentor,
    val_loader,
    device,
    reliability_A=None,
    reliability_B=None,
    num_classes: int = 3,
    metrics_path: str | Path | None = None,
    iteration: int | None = None,
) -> dict:
    was_training = model.training
    model.eval()
    rows = {"unet": [], "vnet": [], "fused": []}
    class_rows = []
    equal_weight_fallback = reliability_A is None or reliability_B is None
    for batch in val_loader:
        image = batch["image"].to(device)
        label = batch["mask"]
        out = model(image, return_features=False)
        logits_a = out["logits_a"]
        logits_b = out["logits_b"]
        fused = _fused_logits(logits_a, logits_b, reliability_A, reliability_B)
        pred_a = torch.argmax(logits_a, dim=1)
        pred_b = torch.argmax(logits_b, dim=1)
        pred_f = torch.argmax(fused, dim=1)
        for key, pred in (("unet", pred_a), ("vnet", pred_b), ("fused", pred_f)):
            rows[key].append(
                multiclass_segmentation_metrics(
                    label.squeeze(0).detach().cpu().numpy(),
                    pred.squeeze(0).detach().cpu().numpy(),
                    num_classes=num_classes,
                )
            )
        class_rows.append(rows["fused"][-1])

    if not rows["fused"]:
        raise ValueError("Validation loader is empty")

    result = {
        "iteration": iteration,
        "equal_weight_fallback": equal_weight_fallback,
        "unet_avg_dice": safe_nanmean([r["avg_dice"] for r in rows["unet"]]),
        "vnet_avg_dice": safe_nanmean([r["avg_dice"] for r in rows["vnet"]]),
        "fused_avg_dice": safe_nanmean([r["avg_dice"] for r in rows["fused"]]),
        "fused_avg_iou": safe_nanmean([r["avg_iou"] for r in rows["fused"]]),
        "fused_avg_hd95": safe_nanmean([r["avg_hd95"] for r in rows["fused"]]),
    }
    for cls in range(1, num_classes):
        result[f"class_{cls}_dice"] = safe_nanmean([r[f"class_{cls}_dice"] for r in class_rows])
        result[f"class_{cls}_iou"] = safe_nanmean([r[f"class_{cls}_iou"] for r in class_rows])
        result[f"class_{cls}_hd95"] = safe_nanmean([r[f"class_{cls}_hd95"] for r in class_rows])

    if metrics_path is not None:
        metrics_path = Path(metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
    if was_training:
        model.train()
    return result
