"""Inspect SAGE-SAM/KnowSAM checkpoint keys."""

from __future__ import annotations

import argparse
import json

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu")
    state = payload.get("model", payload)
    print(json.dumps({"num_keys": len(state), "first_keys": list(state)[:20]}, indent=2))


if __name__ == "__main__":
    main()
