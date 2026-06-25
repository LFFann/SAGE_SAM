"""Checkpoint helpers for SAGE-SAM."""

from __future__ import annotations

from pathlib import Path

import torch


def _safe_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
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
