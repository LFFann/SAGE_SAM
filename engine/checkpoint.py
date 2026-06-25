"""Checkpoint helpers for SAGE-SAM."""

from __future__ import annotations

from pathlib import Path
import os
import random
from datetime import datetime, timezone

import numpy as np
import torch


def _safe_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_checkpoint(path: str | Path, model, optimizer=None, step: int = 0, calibration: dict | None = None, split_manifest: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "step": step,
        "calibration": calibration or {},
        "split_manifest": split_manifest or {},
        "rng_state": torch.get_rng_state(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, model, optimizer=None, strict: bool = False) -> dict:
    payload = _safe_load(path, map_location="cpu")
    model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if "rng_state" in payload:
        torch.set_rng_state(payload["rng_state"])
    return payload


def convert_knowsam_state_dict(state_dict: dict) -> dict:
    converted = {}
    skipped = []
    for key, value in state_dict.items():
        if key.startswith("Discriminator.") or key.startswith("sam.") or "mask_decoder" in key or "prompt_encoder" in key:
            skipped.append(key)
            continue
        converted[key] = value
    return {"state_dict": converted, "skipped": skipped}


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def json_safe_args(args) -> dict:
    if args is None:
        return {}
    items = vars(args) if hasattr(args, "__dict__") else dict(args)
    return {key: str(value) if isinstance(value, Path) else value for key, value in items.items()}


class CheckpointManager:
    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _atomic_save(self, payload: dict, path: Path) -> Path:
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        return path

    def build_payload(
        self,
        *,
        iteration: int,
        best_metric: float,
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        args=None,
        dataset_root: str = "",
        dataset_fingerprint: str = "",
        calibration_manifest: dict | None = None,
        semantic_calibration_state: dict | None = None,
        structure_calibration_state: dict | None = None,
        class_weights=None,
        structure_cache_metadata: dict | None = None,
        git_commit: str | None = None,
    ) -> dict:
        return {
            "format_version": 1,
            "method": "SAGE-SAM",
            "iteration": int(iteration),
            "best_metric": float(best_metric),
            "best_metric_name": "fused_avg_dice",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "args": json_safe_args(args),
            "dataset_root": str(dataset_root),
            "dataset_fingerprint": dataset_fingerprint,
            "calibration_manifest": calibration_manifest or {},
            "semantic_calibration_state": semantic_calibration_state or {},
            "structure_calibration_state": structure_calibration_state or {},
            "class_weights": class_weights.tolist() if torch.is_tensor(class_weights) else class_weights,
            "structure_cache_metadata": structure_cache_metadata or {},
            "rng_state": capture_rng_state(),
            "git_commit": git_commit,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def save(self, filename: str, payload: dict) -> Path:
        return self._atomic_save(payload, self.checkpoint_dir / filename)

    def save_latest(self, payload: dict) -> Path:
        return self.save("latest.pth", payload)

    def save_best(self, payload: dict) -> Path:
        return self.save("best_fused_dice.pth", payload)

    def save_final(self, payload: dict) -> Path:
        return self.save("final.pth", payload)

    def load(self, path: str | Path, model, optimizer=None, scheduler=None, scaler=None, strict: bool = False) -> dict:
        payload = _safe_load(path, map_location="cpu")
        if payload.get("method") not in {None, "SAGE-SAM"}:
            raise ValueError(f"Unsupported checkpoint method: {payload.get('method')}")
        incompatible = []
        model_state = payload.get("model", payload)
        current_state = model.state_dict()
        filtered = {}
        for key, value in model_state.items():
            if key in current_state and tuple(current_state[key].shape) == tuple(value.shape):
                filtered[key] = value
            elif key in current_state:
                incompatible.append((key, tuple(value.shape), tuple(current_state[key].shape)))
        if incompatible:
            raise ValueError(f"Checkpoint shape mismatch: {incompatible[:10]}")
        report = model.load_state_dict(filtered, strict=strict)
        if optimizer is not None and payload.get("optimizer"):
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload.get("scheduler"):
            scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None and payload.get("scaler"):
            scaler.load_state_dict(payload["scaler"])
        restore_rng_state(payload.get("rng_state"))
        payload["load_report"] = {
            "loaded_keys": len(filtered),
            "missing_keys": list(report.missing_keys),
            "unexpected_keys": list(report.unexpected_keys),
            "shape_mismatch": incompatible,
        }
        return payload
