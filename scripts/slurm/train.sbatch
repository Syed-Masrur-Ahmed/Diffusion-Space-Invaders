#!/bin/bash
# Training job on Discovery (ROADMAP 4.15). Partition, GPU type, time limit and account are
# passed on the sbatch command line, never written here. Run from the repo root.
#
#   CONFIG=configs/train/debug.yaml sbatch --partition=<gpu-partition> --time=<HH:MM:SS> \
#       scripts/slurm/train.sbatch --set data.root=<scratch>/data
#
# Chain a follow-up job that auto-resumes the same run (repeat for longer runs):
#   jid=$(CONFIG=configs/train/debug.yaml sbatch --parsable --partition=... --time=... scripts/slurm/train.sbatch)
#   CONFIG=configs/train/debug.yaml sbatch --dependency=afterany:$jid --partition=... --time=... scripts/slurm/train.sbatch
#
# Extra arguments after the script name go to train.py (for example --set overrides).
# Slurm sends SIGUSR1 to this batch shell 300 s before the time limit. Because of `exec`,
# the shell *is* the Python trainer, which saves ckpt_last.pt and exits 0.
#SBATCH --job-name=wm-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --signal=B:USR1@300
#SBATCH --output=runs/slurm-%j.out

set -euo pipefail
: "${CONFIG:?set CONFIG=configs/train/<name>.yaml}"
cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
echo "job ${SLURM_JOB_ID:-local} on $(hostname), config $CONFIG, args: $*"
exec python scripts/train.py --config "$CONFIG" --resume auto "$@"
