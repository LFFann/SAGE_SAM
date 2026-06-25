"""Create a train/calibration split from sample IDs."""

from __future__ import annotations

import argparse

from dataloader.calibration_split import make_calibration_split, save_split_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ids", nargs="+")
    parser.add_argument("--output", default="calibration_split.json")
    parser.add_argument("--fraction", type=float, default=0.15)
    args = parser.parse_args()
    save_split_manifest(make_calibration_split(args.ids, fraction=args.fraction), args.output)
    print(args.output)


if __name__ == "__main__":
    main()
