#!/bin/bash
# Re-runs only Integrated Gradients at more integration steps for all 12 arch x
# dataset combinations. ig_convergence.py showed a 11-53% median completeness
# error at the original 20 steps; at 200 steps it is at most 2% on every
# checkpoint tested (analysis/ig_convergence_sweep.json). Same seed and image
# count as the main run, so run_experiment.py selects exactly the same images.
# Output goes to a separate results_ig<steps>/ tree, so the original per-image
# results are never touched; merge_ig_rerun.py then swaps the IG entries in.
#
# Run from the repository root: bash jobs/submit_ig_rerun.sh [steps]
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
STEPS="${1:-200}"

for arch in vit_b16 vit_l16 deit_b16; do
  for dataset in imagenette imagenet cifar10 cifar100; do
    echo "submit: IG@${STEPS} $arch/$dataset"
    sbatch --job-name="saco_ig${STEPS}_${arch}_${dataset}" \
      --output="logs/saco_ig${STEPS}_${arch}_${dataset}_%j.out" \
      --error="logs/saco_ig${STEPS}_${arch}_${dataset}_%j.err" \
      --export="ARCH=${arch},DATASET=${dataset},NUM_IMAGES=1000,IG_STEPS=${STEPS},IG_INTERNAL_BATCH=25,METHODS=IG,OUTPUT_DIR=$(pwd)/results_ig${STEPS}" \
      jobs/run_experiment.job
  done
done
