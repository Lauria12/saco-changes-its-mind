#!/bin/bash
# Submits the register-token F-Fidelity pilot: DeiT-B/16 robustified against
# blur/noise/shuffle on both CIFAR-10 and CIFAR-100 -- the two checkpoints where
# Attention Rollout collapses onto a content-independent "register" sink token
# (see the paper's register-token section).
#
# This is a falsification test for the register-token account: unlike the earlier
# vit_b16/imagenette pilot (which found a real OOD-artifact effect for blur),
# robustifying against a perturbation is predicted NOT to fix a register-token
# failure, since it isn't plausibly an OOD reaction in the first place.
#
# For each of the 6 (dataset, strategy) combinations this submits two chained
# jobs: a fine-tune job (finetune_perturbation_robust.py), then a full SaCo
# re-evaluation job (run_experiment.job) that only starts once the fine-tune
# succeeds (--dependency=afterok), so you can fire-and-forget the whole pilot.
#
# Run from the repository root: bash jobs/submit_registers_pilot.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # repository root
mkdir -p logs

SUBMIT_ROOT="$(pwd)"
CKPT_DIR="${SUBMIT_ROOT}/checkpoints/robust_registers"

for dataset in cifar10 cifar100; do
  for strat in blur noise shuffle; do
    echo "submit: finetune deit_b16/${dataset}/${strat}"
    FT_JOBID=$(sbatch --parsable --job-name="saco_reg_ft_${dataset}_${strat}" \
      --partition=gpu_a100 --gpus=1 --cpus-per-task=8 --time=01:30:00 \
      --account=YOUR_SLURM_ACCOUNT \
      --output="logs/saco_reg_ft_${dataset}_${strat}_%j.out" \
      --error="logs/saco_reg_ft_${dataset}_${strat}_%j.err" \
      --wrap="module purge; module load 2024; module load Anaconda3/2024.06-1; source activate saco; export LD_LIBRARY_PATH=\"\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}\"; cd src; python finetune_perturbation_robust.py --arch deit_b16 --dataset ${dataset} --strategy ${strat} --data-root ../data --output-dir ${CKPT_DIR} --n-train-images 1000 --epochs 3 --log-every 100")
    echo "  -> fine-tune job ${FT_JOBID}"

    CHECKPOINT_PATH="${CKPT_DIR}/deit_b16_${dataset}_${strat}_robust.pt"
    OUTPUT_DIR="${SUBMIT_ROOT}/results_robust_registers/${strat}/deit_b16_${dataset}"

    echo "submit: eval deit_b16/${dataset}/${strat} (waits on ${FT_JOBID})"
    sbatch --dependency=afterok:${FT_JOBID} \
      --job-name="saco_reg_eval_${dataset}_${strat}" \
      --output="logs/saco_reg_eval_${dataset}_${strat}_%j.out" \
      --error="logs/saco_reg_eval_${dataset}_${strat}_%j.err" \
      --export="ARCH=deit_b16,DATASET=${dataset},CHECKPOINT=${CHECKPOINT_PATH},OUTPUT_DIR=${OUTPUT_DIR},NUM_IMAGES=1000" \
      jobs/run_experiment.job
  done
done

echo ""
echo "All 12 jobs submitted (6 fine-tune + 6 dependency-chained eval)."
echo "Check status with: squeue -u \$USER"
echo "Results will land in: ${SUBMIT_ROOT}/results_robust_registers/{blur,noise,shuffle}/deit_b16_{cifar10,cifar100}/results.json"
