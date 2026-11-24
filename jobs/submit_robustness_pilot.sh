#!/bin/bash
# Submits the F-Fidelity-style perturbation-robustness fine-tuning pilot:
# vit_b16/imagenette robustified separately against blur/noise/shuffle (the 3
# strategies where Raw Attention flips to below-random in the original eval).
# Run from the repository root: bash jobs/submit_robustness_pilot.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

for strat in blur noise shuffle; do
  echo "submit: robust_${strat}"
  sbatch --job-name="saco_robust_${strat}" \
    --partition=gpu_a100 --gpus=1 --cpus-per-task=8 --time=01:30:00 \
    --account=YOUR_SLURM_ACCOUNT \
    --output="logs/saco_robust_${strat}_%j.out" \
    --error="logs/saco_robust_${strat}_%j.err" \
    --wrap="module purge; module load 2024; module load Anaconda3/2024.06-1; source activate saco; export LD_LIBRARY_PATH=\"\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}\"; cd src; python finetune_perturbation_robust.py --arch vit_b16 --dataset imagenette --strategy ${strat} --data-root ../data --output-dir ../checkpoints/robust --n-train-images 1000 --epochs 3 --log-every 100"
done
