"""Calibration split helpers that never borrow validation or test samples."""

from __future__ import annotations

import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


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


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _stable_sample_id(path: Path, dataset_root: Path) -> str:
    return path.relative_to(dataset_root).as_posix()


def _mask_path_for(image_path: Path, labeled_image_dir: Path, labeled_mask_dir: Path) -> Path:
    rel = image_path.relative_to(labeled_image_dir)
    direct = labeled_mask_dir / rel
    if direct.exists():
        return direct
    for ext in IMAGE_EXTENSIONS:
        candidate = (labeled_mask_dir / rel).with_suffix(ext)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing mask for labeled image: {image_path}")


def collect_labeled_samples(labeled_image_dir: Path, labeled_mask_dir: Path, dataset_root: Path) -> list[dict]:
    samples = []
    for image_path in sorted(labeled_image_dir.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            mask_path = _mask_path_for(image_path, labeled_image_dir, labeled_mask_dir)
            samples.append(
                {
                    "sample_id": _stable_sample_id(image_path, dataset_root),
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                }
            )
    if not samples:
        raise ValueError(f"No labeled images found in {labeled_image_dir}")
    return samples


def dataset_fingerprint_from_paths(paths: Sequence[Path], dataset_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.as_posix()):
        stat = path.stat()
        rel = path.relative_to(dataset_root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(int(stat.st_mtime_ns)).encode("ascii"))
    return digest.hexdigest()


def compute_dataset_fingerprint(dataset_root: Path) -> str:
    files = [
        path
        for path in dataset_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not any(part in {"val", "test"} for part in path.relative_to(dataset_root).parts[:1])
    ]
    if not files:
        raise ValueError(f"No dataset image/mask files found under {dataset_root}")
    return dataset_fingerprint_from_paths(files, dataset_root)


def _class_presence(mask_path: Path, num_classes: int) -> list[int]:
    mask = np.asarray(Image.open(mask_path).convert("L"))
    values = set(int(v) for v in np.unique(mask))
    illegal = sorted(v for v in values if v < 0 or v >= num_classes)
    if illegal:
        raise ValueError(f"Mask {mask_path} contains illegal class values {illegal}; expected 0..{num_classes - 1}")
    return [int(cls in values) for cls in range(num_classes)]


def _group_id(sample_id: str, group_regex: str | None) -> str:
    if not group_regex:
        return sample_id
    match = re.search(group_regex, sample_id)
    return match.group(1) if match and match.groups() else (match.group(0) if match else sample_id)


def load_or_create_calibration_manifest(
    labeled_image_dir: Path,
    labeled_mask_dir: Path,
    output_path: Path,
    ratio: float,
    seed: int,
    num_classes: int,
    group_regex: str | None,
    dataset_fingerprint: str,
) -> dict:
    output_path = Path(output_path)
    if output_path.exists():
        manifest = json.loads(output_path.read_text(encoding="utf-8"))
        if manifest.get("dataset_fingerprint") != dataset_fingerprint:
            raise ValueError(
                "Calibration manifest dataset_fingerprint mismatch: "
                f"{manifest.get('dataset_fingerprint')} != {dataset_fingerprint}"
            )
        return manifest

    if not 0.0 < ratio < 1.0:
        raise ValueError("calibration_ratio must be in (0, 1)")

    dataset_root = labeled_image_dir.parents[1]
    samples = collect_labeled_samples(labeled_image_dir, labeled_mask_dir, dataset_root)
    presence = {}
    class_presence_counts = {str(cls): 0 for cls in range(num_classes)}
    for sample in samples:
        mask_presence = _class_presence(Path(sample["mask_path"]), num_classes)
        presence[sample["sample_id"]] = mask_presence
        for cls, flag in enumerate(mask_presence):
            class_presence_counts[str(cls)] += int(flag)

    missing_foreground = [cls for cls in range(1, num_classes) if class_presence_counts[str(cls)] == 0]
    if missing_foreground:
        raise ValueError(f"Calibration split cannot cover missing foreground classes {missing_foreground}; counts={class_presence_counts}")

    groups: dict[str, list[str]] = {}
    for sample in samples:
        groups.setdefault(_group_id(sample["sample_id"], group_regex), []).append(sample["sample_id"])
    group_keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(group_keys)
    target_count = max(1, int(round(len(samples) * ratio)))

    calibration_ids: list[str] = []
    covered = {cls: 0 for cls in range(1, num_classes)}
    for group in group_keys:
        if len(calibration_ids) >= target_count and all(v > 0 for v in covered.values()):
            break
        for sample_id in groups[group]:
            calibration_ids.append(sample_id)
            for cls in covered:
                covered[cls] += presence[sample_id][cls]

    supervised_ids = sorted({sample["sample_id"] for sample in samples} - set(calibration_ids))
    calibration_ids = sorted(set(calibration_ids))
    if not supervised_ids:
        raise ValueError("Calibration split consumed all labeled samples; lower calibration_ratio")
    if set(supervised_ids) & set(calibration_ids):
        raise AssertionError("supervised_ids and calibration_ids overlap")

    manifest = {
        "version": 1,
        "seed": seed,
        "calibration_ratio": ratio,
        "group_regex": group_regex,
        "warning": None if group_regex else "No group_regex provided; sample-level calibration split was used.",
        "dataset_fingerprint": dataset_fingerprint,
        "supervised_ids": supervised_ids,
        "calibration_ids": calibration_ids,
        "class_presence_counts": class_presence_counts,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
