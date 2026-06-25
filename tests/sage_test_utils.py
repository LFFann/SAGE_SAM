from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from dataloader.calibration_split import compute_dataset_fingerprint
from sage_ssl.structure_cache import save_structure_cache, tensor_hash, write_cache_manifest


def _draw_sample(size: int, idx: int) -> tuple[Image.Image, Image.Image]:
    image = Image.new("RGB", (size, size), (20, 20, 20))
    mask = Image.new("L", (size, size), 0)
    draw_i = ImageDraw.Draw(image)
    draw_m = ImageDraw.Draw(mask)
    off = 3 + idx % 5
    draw_i.ellipse((off, off, size // 2, size // 2), fill=(180, 50, 50))
    draw_m.ellipse((off, off, size // 2, size // 2), fill=1)
    draw_i.rectangle((size // 2 - 2, size // 2 - 2, size - off, size - off), fill=(50, 180, 90))
    draw_m.rectangle((size // 2 - 2, size // 2 - 2, size - off, size - off), fill=2)
    return image, mask


def create_synthetic_real_dataset(root: Path, dataset_name: str = "synthetic_3class", size: int = 32) -> Path:
    dataset_root = root / dataset_name
    counts = {"labeled": 6, "unlabeled": 4, "val": 2, "test": 2}
    for split, count in counts.items():
        (dataset_root / split / "image").mkdir(parents=True, exist_ok=True)
        if split != "unlabeled":
            (dataset_root / split / "mask").mkdir(parents=True, exist_ok=True)
        for idx in range(count):
            image, mask = _draw_sample(size, idx)
            image.save(dataset_root / split / "image" / f"{split}_{idx:03d}.png")
            if split != "unlabeled":
                mask.save(dataset_root / split / "mask" / f"{split}_{idx:03d}.png")
    return dataset_root


def sample_ids_for_cache(dataset_root: Path) -> list[str]:
    ids = []
    for split in ("labeled", "unlabeled"):
        for path in sorted((dataset_root / split / "image").glob("*.png")):
            ids.append(path.relative_to(dataset_root).as_posix())
    return ids


def create_synthetic_structure_cache(dataset_root: Path, cache_root: Path, grid_size: int = 32) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    ids = sample_ids_for_cache(dataset_root)
    for idx, sample_id in enumerate(ids):
        gen = torch.Generator().manual_seed(idx)
        emb = torch.randn(1, 4, grid_size, grid_size, generator=gen)
        save_structure_cache(cache_root, sample_id, emb, tensor_hash(emb))
    metadata = {
        "dataset": dataset_root.name,
        "dataset_root": str(dataset_root),
        "dataset_fingerprint": compute_dataset_fingerprint(dataset_root),
        "grid_size": grid_size,
        "sample_count": len(ids),
        "sample_ids": ids,
        "uses_real_sam": False,
    }
    write_cache_manifest(cache_root, metadata)
    return cache_root
