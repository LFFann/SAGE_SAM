"""Convert copied KnowSAM checkpoints by dropping inactive SAGE-SAM modules."""

from __future__ import annotations

import argparse

import torch

from engine.checkpoint import convert_knowsam_state_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    payload = torch.load(args.input, map_location="cpu")
    state = payload.get("model", payload)
    converted = convert_knowsam_state_dict(state)
    torch.save({"model": converted["state_dict"], "skipped": converted["skipped"]}, args.output)
    print(f"saved {args.output}; skipped {len(converted['skipped'])} inactive keys")


if __name__ == "__main__":
    main()
