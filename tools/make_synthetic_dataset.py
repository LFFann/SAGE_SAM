"""Create a tiny three-class synthetic segmentation dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _sample(size: int, idx: int) -> tuple[Image.Image, Image.Image]:
    image = Image.new("RGB", (size, size), (18, 18, 18))
    mask = Image.new("L", (size, size), 0)
    draw_i = ImageDraw.Draw(image)
    draw_m = ImageDraw.Draw(mask)
    offset = 4 + idx % 7
    draw_i.ellipse((offset, offset, size // 2, size // 2), fill=(180, 60, 60))
    draw_m.ellipse((offset, offset, size // 2, size // 2), fill=1)
    draw_i.rectangle((size // 2 - 3, size // 2 - 3, size - offset, size - offset), fill=(60, 170, 90))
    draw_m.rectangle((size // 2 - 3, size // 2 - 3, size - offset, size - offset), fill=2)
    arr = np.asarray(image).astype(np.uint8)
    arr = np.clip(arr + np.random.default_rng(idx).integers(0, 18, arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(arr), mask


def make_dataset(root: str | Path, size: int = 64) -> Path:
    root = Path(root)
    splits = {"labeled": 4, "unlabeled": 4, "calibration": 2, "val": 2, "test": 2}
    for split, count in splits.items():
        (root / split / "image").mkdir(parents=True, exist_ok=True)
        if split != "unlabeled":
            (root / split / "mask").mkdir(parents=True, exist_ok=True)
        for idx in range(count):
            image, mask = _sample(size, idx)
            image.save(root / split / "image" / f"{split}_{idx:03d}.png")
            if split != "unlabeled":
                mask.save(root / split / "mask" / f"{split}_{idx:03d}.png")
    return root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/synthetic_dataset")
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()
    print(make_dataset(args.root, args.size))


if __name__ == "__main__":
    main()
