"""Precompute frozen SAM image embeddings for structure graphs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Model.sam_structure_encoder import SAMStructureEncoder
from dataloader.builders import SAGEImageDataset, resolve_dataset_root, validate_dataset_layout
from dataloader.calibration_split import compute_dataset_fingerprint
from sage_ssl.structure_cache import save_structure_cache, tensor_hash, write_cache_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "--data-path", dest="data_path")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_cache", "--output-cache", "--output", dest="output_cache", required=True)
    parser.add_argument("--sam_checkpoint", "--sam-checkpoint", dest="sam_checkpoint")
    parser.add_argument("--model_type", "--model-type", dest="model_type", default="vit_b")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image_size", "--image-size", dest="image_size", type=int, default=256)
    parser.add_argument("--num_classes", "--num-classes", dest="num_classes", type=int, default=3)
    parser.add_argument("--in_channels", "--in-channels", dest="in_channels", type=int, default=3)
    parser.add_argument("--point_nums", "--point-nums", dest="point_nums", type=int, default=5)
    parser.add_argument("--box_nums", "--box-nums", dest="box_nums", type=int, default=1)
    parser.add_argument("--mod", default="seg")
    parser.add_argument("--thd", action="store_true")
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--structure_grid_size", "--structure-grid-size", dest="structure_grid_size", type=int, default=32)
    parser.add_argument("--synthetic", action="store_true", help="Use deterministic synthetic embeddings instead of a real SAM checkpoint.")
    args = parser.parse_args()

    if not args.data_path:
        raise SystemExit("--data_path is required")
    dataset_root = resolve_dataset_root(args.data_path, args.dataset)
    validate_dataset_layout(dataset_root)
    cache_root = Path(args.output_cache)
    cache_root.mkdir(parents=True, exist_ok=True)
    encoder = SAMStructureEncoder(
        checkpoint=None if args.synthetic else args.sam_checkpoint,
        model_type=args.model_type,
        device=args.device,
        image_size=args.image_size,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        point_nums=args.point_nums,
        box_nums=args.box_nums,
        mod=args.mod,
        thd=args.thd,
        chunk=args.chunk,
    )
    sample_ids = []
    for split, has_mask in (("labeled", True), ("unlabeled", False)):
        dataset = SAGEImageDataset(
            dataset_root,
            split,
            num_classes=args.num_classes,
            image_size=args.image_size,
            has_mask=has_mask,
            in_channels=args.in_channels,
        )
        for sample in dataset:
            image = sample["image"].unsqueeze(0).to(args.device)
            embedding = encoder(image, output_size=(args.structure_grid_size, args.structure_grid_size))
            save_structure_cache(cache_root, sample["sample_id"], embedding.cpu(), tensor_hash(image.cpu()))
            sample_ids.append(sample["sample_id"])
    metadata = {
        "dataset": args.dataset,
        "dataset_root": str(dataset_root),
        "dataset_fingerprint": compute_dataset_fingerprint(dataset_root),
        "grid_size": args.structure_grid_size,
        "sample_count": len(sample_ids),
        "sample_ids": sorted(sample_ids),
        "uses_real_sam": bool(args.sam_checkpoint and not args.synthetic),
    }
    write_cache_manifest(cache_root, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
