# cp5-B: Full dataset and final training run

## Summary
- `configs/data/full.yaml`: 5M steps at 84/max, four jobs: agent ε=0.05 (20%, new, for late-wave and new-wave frames), agent ε=0.2 (40%), agent ε=0.3 (20%), random (20%). `base_seed` 3e8, so no episode repeats a debug seed. Expected size ~7.5 GB.
- `configs/train/final.yaml`: `base` U-Net (34,352,387 parameters), K=4, `data.dirs: [full, human]`, batch 128, 200k steps, `aug_max` 0.7 shared (D22), action dropout 0.1, sampler Euler N=2 (D11), checkpoints every 10k (keep 5). Otherwise identical to `debug.yaml`.
- `scripts/slurm/final_chain.sh`: submits N chained `train.sbatch` jobs (`afterany`, each auto-resumes). `DRY_RUN=1` prints the commands.
- **Not started yet:** collection, validation, coverage report, and the training run all need Discovery (Syed). Commands below.

## Files
Created: `configs/data/full.yaml`, `configs/train/final.yaml`, `scripts/slurm/final_chain.sh`, `tests/test_fullrun.py`, `progress/cp5-B.md`.

## Tests
`tests/test_fullrun.py` (9, fast suite):
- `full.yaml` plans exactly `total_steps` with 1, 8 and 16 workers (the collector's own `plan_parts` validation, including the seed-range check); 2M ≤ steps ≤ 5M; 84/max; `out_dir` `data/full`.
- It has an agent job with ε ≤ 0.05 and ≥ 15% random play.
- Its seeds (up to 16 workers) stay below `EVAL_SEED_BASE` and above every debug seed for debug collected with 1, 4 or 10 workers.
- `final.yaml` has exactly the keys of `debug.yaml`; name `final`, dirs `[full, human]`, K=4, batch in 64–128, `aug_max` 0.7 shared, `p_action_drop` 0.1, `save_every` 10k, Euler N=2, and the same `edm`/`sampler` settings as `debug.yaml`.
- Its model is `configs/model/base.yaml`'s 84 channel list and builds a 34.0–35.0M-parameter U-Net.
- `final_chain.sh` with `DRY_RUN=1`: 3 jobs, no dependency on the first, `afterany` on the previous for the rest, `--set data.root=...` before extra args; fails without `DATA_ROOT`.

Commands (macOS arm64, M3, Python 3.13.2, CPU):
- `pytest -m "not slow and not gpu and not network"` → 297 passed, 8 deselected in 44 s (288 at `cp4` + 9 new).
- Dry run: `python scripts/train.py --config configs/train/final.yaml --dry-run --set 'data.dirs=[debug,human]' --set data.batch_size=8 --set train.val_windows=64` → `UNet: 34,352,387 parameters`, `dry run: 5 steps, last loss 2.4916, val loss 2.4227, samples (8, 3, 84, 84)` in 90 s. `data/human/` does not exist on this machine, so this also shows a missing human dir is fine (shards are listed per dir; `data/debug` stands in for `data/full`).

## Results
None yet (see Discovery commands).

Size and time estimates:
- Disk: `data/debug` is 740 MiB for 500,688 rows, ~1.5 GB per million rows (the ROADMAP guessed 4–8 GB). 5M rows ≈ 7.5 GB.
- Collection: `data/debug` took ~15 min of worker time on 4 workers (CP2-A), so 5M is ~2.5 h on 4 workers or ~40 min on 16.
- Training: `small` ran at ~385 samples/s on an L40S (CP4-B). `base` has 2.4× the parameters; guess 150–200 samples/s, so 200k steps × 128 = 25.6M samples is ~36–47 h. The default chain (3 jobs × 2 days) covers that with room to spare. Check `throughput` in the first job's log and set `train.steps` from it.

## Discovery commands (for Syed)
From the repo root on `cp5/B-fullrun` (or `main` after CP5), venv active. `S=/dartfs-hpc/scratch/f0079km/wm/data`.

1. **Agent weights on a login node** (compute nodes may have no internet): `python -c "from wm.agents.pretrained import ZooAgentPolicy; ZooAgentPolicy()"`.
2. **Collect** (CPU only; uses no GPU). On a CPU partition with 16 cores:
   ```bash
   sbatch -p <cpu-partition> -c 16 -t 06:00:00 -o runs/slurm-%j.out \
     --wrap "source .venv/bin/activate && python scripts/collect.py --config configs/data/full.yaml --workers 16 --set out_dir=$S/full"
   ```
   Use at most 16 workers (seed range). A rerun with the same `--workers` resumes finished parts. If there is no CPU partition, the same `--wrap` on `gpuq` works but holds a GPU.
3. **Validate and coverage:**
   ```bash
   python scripts/validate_data.py $S/full
   python scripts/coverage_report.py $S/full --out-dir runs/coverage_full
   ```
   `coverage_report.py` names its files `cp2_coverage_*`, so give it its own `--out-dir` (the default would overwrite the CP2 figures). Targets: wave transitions ≥ 50 (D17), the other four as in CP2.
4. **Human data** (optional, any time before training starts): copy everyone's `data/human/` shards into `$S/human/`.
5. **Dry run, then the chain:**
   ```bash
   python scripts/train.py --config configs/train/final.yaml --dry-run --set data.root=$S
   DATA_ROOT=$S bash scripts/slurm/final_chain.sh
   ```
   Defaults: 3 jobs × `2-00:00:00`, `gpuq`, one L40S. Override with `N_JOBS=`, `TIME=`, `GRES=`; extra args go to `train.py` (e.g. `--set train.steps=300000`). Stop only with `scancel --signal=USR1 --batch <id>`.
6. **Monitor:** sample grids in `runs/<stamp>_final/samples/`, `val/loss` in TensorBoard, and every ~50k steps a quick rollout check:
   ```bash
   CMD="python scripts/eval_rollouts.py --config configs/eval/rollouts.yaml --models runs/<stamp>_final/checkpoints/ckpt_last.pt --n 16 --set 'dataset_dirs=[$S/eval]' --set out_dir=runs/eval_rollouts/final_<step>" \
     sbatch -p gpuq --gres=gpu:nvidia_l40s:1 -t 01:00:00 scripts/slurm/eval.sbatch
   ```

## Deviations
- **New test file `tests/test_fullrun.py`.** CP5-B's Owns list has no test file, and its two tests (dry run, validator) need real data. The config and chain-script checks are cheap and catch typos before a multi-day job.
- **Batch 128, 200k steps.** The roadmap says 64–128 and "as many steps as compute allows". 200k is a starting point; adjust from the measured throughput.
- **`keep_last: 5`.** Each `base` checkpoint is ~0.55 GB, so 5 keep 50k steps of history in ~2.75 GB under `runs/`.
- **Collection is not on day 1 of CP5 yet** (needs Discovery or another machine with disk).

## Interface change requests
None.

## Follow-ups
- **Before launching the chain:** have CP5-C run `action_sensitivity.py` on the CP4 aug checkpoint (`--set num_steps=2`). If the model ignores actions and CP5-C adds `action_input: adagn+planes`, `final.yaml` needs that key and the run should start with it rather than restart days in.
- Human play: nobody has recorded yet (target ≥ 1 h). Each person uses their own `first_worker` in `configs/data/human.yaml`.
- Record the measured `base` throughput, the coverage table, and the final checkpoint path here.
