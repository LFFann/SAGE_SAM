#!/usr/bin/env bash
set -euo pipefail
python train_sage_sam.py --device cuda --num-classes 3
