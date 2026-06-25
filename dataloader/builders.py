"""Real-data dataloader construction for SAGE-SAM."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dataloader.calibration_split import IMAGE_EXTENSIONS, compute_dataset_fingerprint
from utils.label_validation import validate_class_index_mask


REQUIRED_LAYOUT = (
    "labeled/image",
    "labeled/mask",
    "unlabeled/image",
    "val/image",
    "val/mask",
    "test/image",
    "test/mask",
)


def resolve_dataset_root(data_path: str | Path, dataset: str) -> Path:
    if not dataset:
        raise ValueError("--dataset is required for real training")
    dataset_name = str(dataset).strip().replace("\\", "/").lstrip("/")
    return (Path(data_path).expanduser() / dataset_name).resolve()


def validate_dataset_layout(dataset_root: Path) -> None:
    missing = [rel for rel in REQUIRED_LAYOUT if not (dataset_root / rel).is_dir()]
    if missing:
        raise FileNotFoundError(f"Dataset root {dataset_root} is missing required directories: {missing}")


def _list_images(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _mask_for_image(image_path: Path, image_dir: Path, mask_dir: Path) -> Path:
    rel = image_path.relative_to(image_dir)
    direct = mask_dir / rel
    if direct.exists():
        return direct
    for ext in IMAGE_EXTENSIONS:
        candidate = (mask_dir / rel).with_suffix(ext)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing mask for {image_path}")


def _sample_id(path: Path, dataset_root: Path) -> str:
    return path.relative_to(dataset_root).as_posix()


def _fingerprint_for_ids(sample_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sorted(sample_ids):
        digest.update(sample_id.encode("utf-8"))
    return digest.hexdigest()


class SAGEImageDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        split: str,
        num_classes: int,
        image_size: int,
        sample_ids: list[str] | None = None,
        has_mask: bool = True,
        in_channels: int = 3,
    ):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.num_classes = num_classes
        self.image_size = image_size
        self.has_mask = has_mask
        self.in_channels = in_channels
        image_dir = self.dataset_root / split / "image"
        mask_dir = self.dataset_root / split / "mask"
        paths = _list_images(image_dir)
        if sample_ids is not None:
            keep = set(sample_ids)
            paths = [path for path in paths if _sample_id(path, self.dataset_root) in keep]
        if not paths:
            raise ValueError(f"No samples found for split={split} under {image_dir}")
        self.records = []
        for image_path in paths:
            record = {
                "image_path": image_path,
                "mask_path": _mask_for_image(image_path, image_dir, mask_dir) if has_mask else None,
                "sample_id": _sample_id(image_path, self.dataset_root),
                "group_id": "",
            }
            self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        array = np.asarray(image).astype(np.float32) / 255.0
        if self.in_channels == 1:
            array = array[..., :1]
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return tensor, tensor.clone()

    def _load_mask(self, path: Path) -> torch.Tensor:
        mask = Image.open(path).convert("L")
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        array = validate_class_index_mask(np.asarray(mask), self.num_classes, source=str(path)).astype(np.int64)
        return torch.from_numpy(array)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image, image_raw = self._load_image(record["image_path"])
        sample = {
            "image": image,
            "image_raw": image_raw,
            "sample_id": record["sample_id"],
            "path": str(record["image_path"]),
            "group_id": record["group_id"],
        }
        if self.has_mask:
            sample["mask"] = self._load_mask(record["mask_path"])
        return sample


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


def _class_weights(dataset: SAGEImageDataset, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for record in dataset.records:
        mask = np.asarray(Image.open(record["mask_path"]).convert("L"))
        validate_class_index_mask(mask, num_classes, source=str(record["mask_path"]))
        unique, class_counts = np.unique(mask, return_counts=True)
        for cls, count in zip(unique, class_counts):
            counts[int(cls)] += int(count)
    weights = counts.sum() / counts.clamp_min(1.0)
    weights = weights / weights.mean().clamp_min(1e-6)
    return weights.float()


def build_sage_dataloaders(
    dataset_root: Path,
    calibration_manifest: dict,
    num_classes: int,
    in_channels: int,
    image_size: int,
    labeled_batch_size: int,
    unlabeled_batch_size: int,
    calibration_batch_size: int,
    num_workers: int,
    seed: int,
) -> dict:
    dataset_root = Path(dataset_root)
    validate_dataset_layout(dataset_root)
    generator = torch.Generator()
    generator.manual_seed(seed)

    supervised = SAGEImageDataset(
        dataset_root,
        "labeled",
        num_classes,
        image_size,
        sample_ids=calibration_manifest["supervised_ids"],
        has_mask=True,
        in_channels=in_channels,
    )
    calibration = SAGEImageDataset(
        dataset_root,
        "labeled",
        num_classes,
        image_size,
        sample_ids=calibration_manifest["calibration_ids"],
        has_mask=True,
        in_channels=in_channels,
    )
    unlabeled = SAGEImageDataset(dataset_root, "unlabeled", num_classes, image_size, has_mask=False, in_channels=in_channels)
    val = SAGEImageDataset(dataset_root, "val", num_classes, image_size, has_mask=True, in_channels=in_channels)
    test = SAGEImageDataset(dataset_root, "test", num_classes, image_size, has_mask=True, in_channels=in_channels)

    return {
        "supervised_loader": DataLoader(
            supervised,
            batch_size=labeled_batch_size,
            shuffle=True,
            drop_last=len(supervised) >= labeled_batch_size,
            num_workers=num_workers,
            pin_memory=False,
            worker_init_fn=seed_worker,
            generator=generator,
        ),
        "unlabeled_loader": DataLoader(
            unlabeled,
            batch_size=unlabeled_batch_size,
            shuffle=True,
            drop_last=len(unlabeled) >= unlabeled_batch_size,
            num_workers=num_workers,
            pin_memory=False,
            worker_init_fn=seed_worker,
            generator=generator,
        ),
        "calibration_loader": DataLoader(calibration, batch_size=calibration_batch_size, shuffle=False, drop_last=False, num_workers=num_workers),
        "val_loader": DataLoader(val, batch_size=1, shuffle=False, drop_last=False, num_workers=num_workers),
        "test_loader": DataLoader(test, batch_size=1, shuffle=False, drop_last=False, num_workers=num_workers),
        "class_weights": _class_weights(supervised, num_classes),
        "dataset_fingerprint": compute_dataset_fingerprint(dataset_root),
        "sample_ids": {
            "supervised": [record["sample_id"] for record in supervised.records],
            "calibration": [record["sample_id"] for record in calibration.records],
            "unlabeled": [record["sample_id"] for record in unlabeled.records],
            "val": [record["sample_id"] for record in val.records],
            "test": [record["sample_id"] for record in test.records],
        },
    }
