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
    path.parent.mkdir(parents=True, exist_ok=True)
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
    path = Path(root) / "metadata.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    legacy_path = Path(root) / "manifest.json"
    legacy_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


class StructureCacheReader:
    def __init__(
        self,
        cache_root: str | Path,
        expected_dataset: str,
        expected_grid_size: int,
        required_sample_ids: set[str],
        expected_dataset_fingerprint: str | None = None,
    ):
        self.cache_root = Path(cache_root)
        self.expected_dataset = expected_dataset
        self.expected_grid_size = expected_grid_size
        self.required_sample_ids = set(required_sample_ids)
        self.expected_dataset_fingerprint = expected_dataset_fingerprint
        self.metadata_path = self.cache_root / "metadata.json"
        self.metadata: dict = {}
        self.embedding_shape: tuple[int, ...] | None = None

    def _sample_path(self, sample_id: str) -> Path:
        return self.cache_root / f"{sample_id}.pt"

    def validate(self) -> dict:
        if not self.cache_root.exists():
            raise FileNotFoundError(self._missing_message([*sorted(self.required_sample_ids)[:20]], len(self.required_sample_ids)))
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Missing structure cache metadata: {self.metadata_path}")
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        grid_size = int(self.metadata.get("grid_size", -1))
        if grid_size != self.expected_grid_size:
            raise ValueError(f"Structure cache grid size mismatch: expected {self.expected_grid_size}, got {grid_size}")
        dataset = self.metadata.get("dataset")
        if dataset not in {self.expected_dataset, None, ""}:
            raise ValueError(f"Structure cache dataset mismatch: expected {self.expected_dataset}, got {dataset}")
        fingerprint = self.metadata.get("dataset_fingerprint")
        if self.expected_dataset_fingerprint and fingerprint and fingerprint != self.expected_dataset_fingerprint:
            raise ValueError(
                "Structure cache dataset_fingerprint mismatch: "
                f"{fingerprint} != {self.expected_dataset_fingerprint}"
            )

        missing = [sample_id for sample_id in sorted(self.required_sample_ids) if not self._sample_path(sample_id).exists()]
        if missing:
            raise FileNotFoundError(self._missing_message(missing[:20], len(missing)))

        seen = set()
        for sample_id in sorted(self.required_sample_ids):
            if sample_id in seen:
                raise ValueError(f"Duplicate structure cache sample_id: {sample_id}")
            seen.add(sample_id)
            embedding = load_structure_cache(self.cache_root, sample_id)
            if not torch.isfinite(embedding).all():
                raise ValueError(f"Non-finite structure embedding for sample_id={sample_id}")
            if embedding.ndim != 4:
                raise ValueError(f"Structure embedding must be [B,D,H,W], got {tuple(embedding.shape)} for {sample_id}")
            if tuple(embedding.shape[-2:]) != (self.expected_grid_size, self.expected_grid_size):
                raise ValueError(f"Structure embedding grid mismatch for {sample_id}: {tuple(embedding.shape[-2:])}")
            if self.embedding_shape is None:
                self.embedding_shape = tuple(embedding.shape)
            elif tuple(embedding.shape) != self.embedding_shape:
                raise ValueError(f"Structure embedding shape mismatch for {sample_id}: {tuple(embedding.shape)} != {self.embedding_shape}")

        return {
            "cache_root": str(self.cache_root),
            "metadata": self.metadata,
            "required_count": len(self.required_sample_ids),
            "embedding_shape": self.embedding_shape,
        }

    def _missing_message(self, sample_ids: list[str], count: int) -> str:
        return (
            f"Structure cache is missing {count} required sample(s); first missing ids={sample_ids}. "
            "Precompute with: python tools/precompute_sam_structure.py "
            f"--data_path \"<data_path>\" --dataset \"{self.expected_dataset}\" --output_cache \"{self.cache_root}\""
        )

    def get(self, sample_ids: list[str], device: torch.device | None = None) -> torch.Tensor:
        embeddings = []
        for sample_id in sample_ids:
            tensor = load_structure_cache(self.cache_root, sample_id)
            embeddings.append(tensor.detach())
        batch = torch.cat(embeddings, dim=0).detach()
        batch.requires_grad_(False)
        if device is not None:
            batch = batch.to(device)
        return batch
