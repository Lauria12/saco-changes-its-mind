#!/bin/bash
# Submits run_experiment.job across the full arch x dataset matrix.
# Usage: ./jobs/submit_matrix.sh [num_images]   (run from the repository root)
#
# Skips CIFAR combos with no registered checkpoint (see
# src/configs/cifar_checkpoints.py) instead of submitting jobs guaranteed to fail
# immediately in load_model().
set -euo pipefail

cd "$(dirname "$0")/.."   # repository root -- sbatch resolves job paths from here

NUM_IMAGES="${1:-1000}"
ARCHES=(vit_b16 vit_l16 deit_b16)
DATASETS=(imagenette imagenet cifar10 cifar100)

mkdir -p logs

has_cifar_checkpoint() {
  local arch="$1" dataset="$2"
  python3 - "$arch" "$dataset" <<'PYEOF'
import sys
sys.path.insert(0, "src")
from configs.cifar_checkpoints import CIFAR_CHECKPOINTS
arch, dataset = sys.argv[1], sys.argv[2]
sys.exit(0 if CIFAR_CHECKPOINTS.get(arch, {}).get(dataset) else 1)
PYEOF
}

for arch in "${ARCHES[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    if [[ "$dataset" == cifar* ]] && ! has_cifar_checkpoint "$arch" "$dataset"; then
      echo "skip: $arch/$dataset -- no checkpoint registered in src/configs/cifar_checkpoints.py"
      continue
    fi
    echo "submit: $arch/$dataset"
    sbatch --job-name="saco_${arch}_${dataset}" \
      --export="ARCH=${arch},DATASET=${dataset},NUM_IMAGES=${NUM_IMAGES}" \
      jobs/run_experiment.job
  done
done
