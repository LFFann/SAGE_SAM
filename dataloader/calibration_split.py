"""Calibration split helpers that never borrow validation or test samples."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence


def make_calibration_split(sample_ids: Sequence[str], fraction: float = 0.15, seed: int = 2026) -> dict[str, list[str]]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    ids = list(sample_ids)
    if not ids:
        raise ValueError("sample_ids is empty")
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_cal = max(1, int(round(len(ids) * fraction)))
    return {"calibration": sorted(ids[:n_cal]), "train": sorted(ids[n_cal:])}


def save_split_manifest(split: dict[str, list[str]], path: str | Path) -> None:
    Path(path).write_text(json.dumps(split, indent=2), encoding="utf-8")
