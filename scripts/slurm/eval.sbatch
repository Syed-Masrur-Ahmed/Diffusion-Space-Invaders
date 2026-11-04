#!/bin/bash
# Evaluation or benchmark job on Discovery (ROADMAP 4.15): runs the command in $CMD on one GPU.
# Partition, GPU type, time limit and account are passed on the sbatch command line. Run from
# the repo root.
#
#   CMD="python scripts/eval_single_step.py --model runs/<run>/checkpoints/ckpt_last.pt --n 2048" \
#       sbatch --partition=<gpu-partition> --time=<HH:MM:SS> scripts/slurm/eval.sbatch
#
# Run it after a training job finishes with --dependency=afterok:<jobid>.
#SBATCH --job-name=wm-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --signal=B:USR1@300
#SBATCH --output=runs/slurm-%j.out

set -euo pipefail
: "${CMD:?set CMD to the command to run, e.g. CMD=\"python scripts/eval_rollouts.py ...\"}"
cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
echo "job ${SLURM_JOB_ID:-local} on $(hostname): $CMD"
exec bash -c "exec $CMD"
