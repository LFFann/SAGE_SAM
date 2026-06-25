#!/usr/bin/env bash
set -euo pipefail
python prediction_sage_sam.py --device cuda --num-classes 3 "$@"
