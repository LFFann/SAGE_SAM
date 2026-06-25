#!/usr/bin/env bash
set -euo pipefail
for cfg in configs/ablation_dual_supervised.yaml configs/ablation_set_only.yaml configs/ablation_set_structure.yaml configs/ablation_full.yaml; do
  echo "Prepared ablation config: ${cfg}"
done
