"""Precompute frozen SAM image embeddings for structure graphs."""

from __future__ import annotations

import argparse

import torch

from Model.sam_structure_encoder import SAMStructureEncoder
from sage_ssl.structure_cache import save_structure_cache, tensor_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="structure_cache")
    parser.add_argument("--sam-checkpoint")
    args = parser.parse_args()
    encoder = SAMStructureEncoder(checkpoint=args.sam_checkpoint)
    image = torch.zeros(1, 3, 64, 64)
    embedding = encoder(image)
    save_structure_cache(args.output, "synthetic", embedding, tensor_hash(image))
    print("structure cache written")


if __name__ == "__main__":
    main()
