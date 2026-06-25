#!/usr/bin/env bash
set -euo pipefail
python tools/precompute_sam_structure.py --output structure_cache "$@"
