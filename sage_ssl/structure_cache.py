"""On-disk cache for frozen SAM structure embeddings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def _safe_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def tensor_hash(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def save_structure_cache(
    root: str | Path,
    sample_id: str,
    embedding: torch.Tensor,
    image_hash: str,
    checkpoint_hash: str = "synthetic",
) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{sample_id}.pt"
    payload = {
        "sample_id": sample_id,
        "image_hash": image_hash,
        "checkpoint_hash": checkpoint_hash,
        "shape": list(embedding.shape),
        "embedding": embedding.detach().cpu().half(),
    }
    torch.save(payload, path)
    return path


def load_structure_cache(root: str | Path, sample_id: str, expected_shape: tuple[int, ...] | None = None) -> torch.Tensor:
    path = Path(root) / f"{sample_id}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing structure cache for sample {sample_id}: {path}")
    payload = _safe_load(path, map_location="cpu")
    embedding = payload["embedding"].float()
    if expected_shape is not None and tuple(embedding.shape) != tuple(expected_shape):
        raise ValueError(f"Structure cache shape mismatch: expected {expected_shape}, got {tuple(embedding.shape)}")
    return embedding


def write_cache_manifest(root: str | Path, manifest: dict) -> Path:
    path = Path(root) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
