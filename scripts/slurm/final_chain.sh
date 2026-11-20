#!/bin/bash
# Submit the CP5-B final run as a chain of train.sbatch jobs (ROADMAP CP5-B, 4.15).
# Each job after the first waits for the previous one (afterany) and resumes ckpt_last.pt;
# a job that starts after the run reached train.steps exits at once. Run from the repo root.
#
#   DATA_ROOT=<scratch>/data bash scripts/slurm/final_chain.sh [extra train.py args, e.g. --set train.steps=300000]
#
# Settings (environment variables):
#   DATA_ROOT  required: dir holding full/ (and human/ if any)
#   N_JOBS     jobs in the chain (default 3)
#   TIME       time limit per job (default 2-00:00:00; gpuq allows up to 5 days)
#   PARTITION  default gpuq
#   GRES       default gpu:nvidia_l40s:1
#   CONFIG     default configs/train/final.yaml
#   DRY_RUN=1  print the sbatch commands instead of submitting
set -euo pipefail
: "${DATA_ROOT:?set DATA_ROOT=<scratch>/data (holds full/ and human/)}"
N_JOBS=${N_JOBS:-3}
TIME=${TIME:-2-00:00:00}
PARTITION=${PARTITION:-gpuq}
GRES=${GRES:-gpu:nvidia_l40s:1}
CONFIG=${CONFIG:-configs/train/final.yaml}
DRY_RUN=${DRY_RUN:-0}

[ "$DRY_RUN" = 1 ] || mkdir -p runs
jid=""
for i in $(seq 1 "$N_JOBS"); do
  dep=()
  [ -n "$jid" ] && dep=(--dependency=afterany:"$jid")
  cmd=(sbatch --parsable --partition="$PARTITION" --gres="$GRES" --time="$TIME" ${dep[@]+"${dep[@]}"}
       scripts/slurm/train.sbatch --set data.root="$DATA_ROOT" "$@")
  if [ "$DRY_RUN" = 1 ]; then
    echo "CONFIG=$CONFIG ${cmd[*]}"
    jid="<job $i>"
  else
    jid=$(CONFIG="$CONFIG" "${cmd[@]}")
    echo "job $i/$N_JOBS: $jid"
  fi
done
