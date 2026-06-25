"""Inspect a cached structure embedding."""

from __future__ import annotations

import argparse
import json

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_file")
    args = parser.parse_args()
    payload = torch.load(args.cache_file, map_location="cpu")
    print(json.dumps({key: value for key, value in payload.items() if key != "embedding"}, indent=2))


if __name__ == "__main__":
    main()
