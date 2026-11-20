#!/bin/bash
# Submit the Milestone 2 jobs on Discovery (notebooks/milestone2.ipynb, E5-E8). Run from the repo root.
#
#   DATA_ROOT=<scratch>/data DRY_RUN=1 bash scripts/slurm/m2_runs.sh all   # print, submit nothing
#   DATA_ROOT=<scratch>/data bash scripts/slurm/m2_runs.sh data            # data/test + data stats (CPU)
#   DATA_ROOT=<scratch>/data bash scripts/slurm/m2_runs.sh subsets         # E6 subset dirs (login node, no job)
#   DATA_ROOT=<scratch>/data bash scripts/slurm/m2_runs.sh train           # R0, E5 placebo, E6, E7, E8 training runs
#   DATA_ROOT=<scratch>/data bash scripts/slurm/m2_runs.sh eval            # E4: copy-last, M1, final on full val, step >= 60
#
# Every training run is configs/train/debug.yaml (small U-Net, batch 64, 50k steps, aug_max 0, seed 0)
# on data/full with exactly one change. R0 has no context noise, so X1 (aug_max 0.7) is one change.
# E6 subsets keep the play-style mix: data/full has 16 writers per job (w = job*16 + i), and
# full_pN links the shards of writers with i < n for every job (n = 2, 4, 8: 12.5/25/50%).
# Links keep the shard names, so each episode stays in the same train/val split (4.5).
#
# Settings: DATA_ROOT (required), PARTITION (gpuq), GRES (gpu:nvidia_l40s:1), CPU_PARTITION (standard),
# TIME_SMALL (06:00:00), TIME_BASE (12:00:00), DRY_RUN=1, and for `eval` M1_CKPT and FINAL_CKPT (the CP3 and
# CP5-B checkpoints, defaults below).
set -euo pipefail
: "${DATA_ROOT:?set DATA_ROOT=<scratch>/data (holds full/ and eval/)}"
PARTITION=${PARTITION:-gpuq}
GRES=${GRES:-gpu:nvidia_l40s:1}
CPU_PARTITION=${CPU_PARTITION:-standard}
TIME_SMALL=${TIME_SMALL:-06:00:00}
TIME_BASE=${TIME_BASE:-12:00:00}
DRY_RUN=${DRY_RUN:-0}
M1_CKPT=${M1_CKPT:-runs/20261105-0558_debug_small/checkpoints/ckpt_last.pt}
FINAL_CKPT=${FINAL_CKPT:-runs/20261118-0501_final/checkpoints/ckpt_last.pt}
CONFIG=configs/train/debug.yaml

run() {  # print or run a command; prints the job id when it submits
  if [ "$DRY_RUN" = 1 ]; then echo "$*" >&2; else "$@"; fi
}

data_jobs() {
  local act="source .venv/bin/activate"
  local jid
  jid=$(run sbatch --parsable -p "$CPU_PARTITION" -c 2 -t 01:00:00 -o runs/slurm-%j.out \
    --wrap "$act && python scripts/collect.py --config configs/data/test.yaml --workers 2 --set out_dir=$DATA_ROOT/test \
            && python scripts/validate_data.py $DATA_ROOT/test")
  [ "$DRY_RUN" = 1 ] && jid="<test job>"
  run sbatch --parsable -p "$CPU_PARTITION" -c 4 -t 04:00:00 -o runs/slurm-%j.out --dependency=afterok:"$jid" \
    --wrap "$act && python scripts/m2_data_stats.py --config $CONFIG --set data.root=$DATA_ROOT \
            --set 'data.dirs=[full]' --holdout eval test --out results/milestone2/data_full"
}

subsets() {
  local n i job d
  for n in 2 4 8; do
    d="$DATA_ROOT/full_p$((n * 100 / 16))"
    run mkdir -p "$d"
    for job in 0 1 2 3; do
      for ((i = 0; i < n; i++)); do
        run bash -c "ln -sf $DATA_ROOT/full/shard_w$(printf %03d $((job * 16 + i)))_*.h5 $d/"
      done
    done
  done
}

train() {  # train <name> <time> [--set ...]: one train.sbatch job on data/full unless a --set overrides data.dirs
  local name=$1 time=$2
  shift 2
  CONFIG=$CONFIG run sbatch --parsable --partition="$PARTITION" --gres="$GRES" --time="$time" \
    scripts/slurm/train.sbatch --set data.root="$DATA_ROOT" --set 'data.dirs=[full]' --set name="m2_$name" "$@"
}

train_jobs() {
  [ "$DRY_RUN" = 1 ] && echo "# each sbatch below runs with CONFIG=$CONFIG" >&2
  [ "$DRY_RUN" = 1 ] || mkdir -p runs
  train r0 "$TIME_SMALL"                                      # R0 = E6 100% = E7 seed 0
  train placebo "$TIME_SMALL" --set train.shuffle_actions=true  # E5: actions shuffled across samples
  train r0_seed1 "$TIME_SMALL" --set seed=1                   # E7
  train r0_seed2 "$TIME_SMALL" --set seed=2                   # E7
  train data_p12 "$TIME_SMALL" --set 'data.dirs=[full_p12]'   # E6 12.5%
  train data_p25 "$TIME_SMALL" --set 'data.dirs=[full_p25]'   # E6 25%
  train data_p50 "$TIME_SMALL" --set 'data.dirs=[full_p50]'   # E6 50%
  train x1_aug "$TIME_SMALL" --set train.aug_max=0.7          # E8 X1
  train x3_sigma093 "$TIME_SMALL" --set edm.sigma_data=0.93   # E8 X3 (from E2)
  train x4_base "$TIME_BASE" --set 'model.ch=[128,256,384]'   # E8 X4 (base width, batch 64 like R0)
}

eval_jobs() {  # one eval.sbatch job per model: eval.sbatch execs CMD, so it cannot chain commands
  local model cmd
  [ "$DRY_RUN" = 1 ] || mkdir -p runs
  for model in copy_last "$M1_CKPT" "$FINAL_CKPT"; do
    cmd="python scripts/eval_single_step.py --config configs/eval/single_step.yaml --model $model --n 2048"
    cmd+=" --set 'data.dataset_dirs=[$DATA_ROOT/full]' --set min_step=60 --set out_dir=results/milestone2/e4"
    [ "$DRY_RUN" = 1 ] && echo "CMD=\"$cmd\" \\" >&2
    CMD=$cmd run sbatch --parsable --partition="$PARTITION" --gres="$GRES" --time=01:00:00 scripts/slurm/eval.sbatch
  done
}

case "${1:-}" in
  data) data_jobs ;;
  subsets) subsets ;;
  train) train_jobs ;;
  eval) eval_jobs ;;
  all) data_jobs; subsets; train_jobs; eval_jobs ;;
  *) echo "usage: $0 data|subsets|train|eval|all" >&2; exit 2 ;;
esac
