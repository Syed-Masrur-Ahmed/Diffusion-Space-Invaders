# cp5-B: Full dataset and final training run

## Summary
- `configs/data/full.yaml`: 5M steps at 84/max, four jobs: agent ε=0.05 (20%, new, for late-wave and new-wave frames), agent ε=0.2 (40%), agent ε=0.3 (20%), random (20%). `base_seed` 3e8, so no episode repeats a debug seed. Expected size ~7.5 GB.
- `configs/train/final.yaml`: `base` U-Net (34,352,387 parameters), K=4, `data.dirs: [full, human]`, batch 128, 200k steps, `aug_max` 0.7 shared (D22), action dropout 0.1, sampler Euler N=2 (D11), checkpoints every 10k (keep 5). Otherwise identical to `debug.yaml`.
- `scripts/slurm/final_chain.sh`: submits N chained `train.sbatch` jobs (`afterany`, each auto-resumes). `DRY_RUN=1` prints the commands.
- **`data/full/` collected and validated on Discovery (2026-11-18):** 5,000,000 steps, 6,355 episodes, 7.0 GiB, all five coverage targets met (wave transitions 488 vs 24 in debug). See Results.
- **Final run finished on Discovery (2026-11-19): 200,000 steps** in ~42 h on one L40S (171 samples/s). Checkpoint for CP6: **`runs/20261118-0501_final/checkpoints/ckpt_last.pt`** (= `ckpt_0200000.pt`, 525 MB, on Discovery only). See Results.

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
**`data/full/`** (Discovery, `/dartfs-hpc/scratch/f0079km/wm/data/full`, partition `standard`, 16 CPU workers, ~14 min: 13,086 worker-seconds / 16). Smoke test first: 2,000 steps on a login node, validator OK.

| job | steps | episodes | mean score | game over | cut by budget |
|---|---:|---:|---:|---:|---:|
| agent ε=0.05 | 1,000,000 | 986 | 725.6 | 970 | 16 |
| agent ε=0.2 | 2,000,000 | 2,261 | 516.6 | 2,245 | 16 |
| agent ε=0.3 | 1,000,000 | 1,285 | 403.9 | 1,269 | 16 |
| random | 1,000,000 | 1,823 | 150.1 | 1,807 | 16 |
| **total** | **5,000,000** | **6,355** | 421.1 | 6,291 | 64 |

- `validate_data.py`: `OK: all invariants hold`. 5,006,355 rows (agent 4,004,532, random 1,001,823, human 0), 80 shards, mean episode 787.8 rows, **7,162 MiB**, compression 14.11 (~1.43 GiB per million rows). Actions: NOOP 11.5%, FIRE 14.8%, RIGHT 13.6%, LEFT 14.5%, RIGHTFIRE 26.4%, LEFTFIRE 19.2%.
- "Cut by budget" is one episode per worker part, truncated when its step budget ran out (`end_reason` truncated); still valid data.
- **Coverage** (`coverage_report.py ... --out-dir runs/coverage_full`, not committed):

  | target | minimum | `data/debug` (500k) | `data/full` (5M) |
  |---|---|---:|---:|
  | frames with ≤ 6 invaders | 5% | 8.48% | **12.15%** |
  | wave transitions | 50 (D17) | 24 | **488** |
  | game overs | 200 | 675 | **6,291** |
  | UFO visible | 1% | 17.51% | **18.35%** |
  | shields ≤ 50% | 10% | 15.26% | **15.70%** |

  Life losses 17,621; max score 2,155. UFO pixel heuristic agrees with RAM[76] on 99.93% of frames. Wave transitions grew 20× for 10× the data: the ε=0.05 job (mean score 726, ~1,000 steps per episode). The wave-start montage shows full formations at score ~630, the value of a cleared wave 1 (36 aliens, 5-30 points per row).

**Do the CP4 models use the actions?** (reason the chain does not wait for CP5-C). Player x (centroid of green pixels in rows 72-79) in the real and generated panels of the four `d11_steps2` rollout videos (aug model, Euler N=2): mean |real - generated| is 0.0-0.3 px over h=1-20 in all four, while the real ship moves over x≈18-54. The ship has no momentum, so tracking its reversals needs the actions. Later gaps: 0.4-5.2 px at h=20-50, 3.6-15 px at h=50-200 (drift). The model does not register hits (generated score stays ~5 while the real one passes 100), which is a dynamics problem (rare, few-pixel events), not action input.

**Final run** (`runs/20261118-0501_final`, Discovery, `final_chain.sh` defaults: 3 jobs × 2 days, `gpuq`, one L40S):
- Started 2026-11-18 ~05:01, wrote `ckpt_0200000.pt` 2026-11-19 23:01: **~42 h for 200k steps**, all inside the first job. Jobs 2 and 3 resumed at 200k and exited at once (`finished at step 200000` in all three logs).
- **Throughput: median 171 samples/s** (estimate was 150–200; `small` was ~385 in CP4-B).
- **Val loss (clean context): 0.0004** at steps 192k–198k, 0.0005 at 200k. CP4's debug models were 0.0012–0.0020 at 50k, but on the `data/debug` val split, so the numbers are not directly comparable. The trainer prints 4 decimals, so the curve looks flat at this scale; the rollout eval is the comparison that matters.
- Kept checkpoints: `ckpt_0160000.pt` … `ckpt_0200000.pt` (every 10k, `keep_last` 5) and `ckpt_last.pt`. Sample grids up to `samples/step_0200000.png`.
- Rollout check on the final checkpoint (`eval_rollouts.py --n 16`, task 4): not run, see Deviations.

Size and time estimates (before collection):
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
4. **Human data:** not for this run (see Deviations). Don't copy anything into `$S/human/` until the chain has finished.
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
- **No human data in the final run.** The roadmap says to include all `data/human/` shards; nobody had recorded yet, and one hour of play is ~54k steps, ~1% of `data/full`. Decision (Syed, 2026-11-18): launch on `full/` only and record human play (Aunonto `first_worker=0`, Michael `first_worker=200`, ≥ 30 min each, full games) **after** the run finishes. Then use it as a human-play test set (`eval_rollouts.py --set 'dataset_dirs=[<scratch>/data/human]'`) and optionally for a short fine-tune. `final.yaml` keeps `dirs: [full, human]`, so nothing may be copied into `<scratch>/data/human` while the chain runs: a resumed job would re-index with the new shards and silently change the data order.
- **New test file `tests/test_fullrun.py`.** CP5-B's Owns list has no test file, and its two tests (dry run, validator) need real data. The config and chain-script checks are cheap and catch typos before a multi-day job.
- **Batch 128, 200k steps.** The roadmap says 64–128 and "as many steps as compute allows". 200k is a starting point; adjust from the measured throughput.
- **`keep_last: 5`.** Each `base` checkpoint is ~0.55 GB, so 5 keep 50k steps of history in ~2.75 GB under `runs/`.
- **No `eval_rollouts.py --n 16` checks (task 4).** Task 4 is monitoring, to catch a bad run early; the run finished cleanly (val loss flat, sample grids to 200k), so a 1 h GPU job on the final checkpoint would only preview CP6, whose eval runs the horizon curves (final vs debug aug-on vs copy-last) and counterfactuals on this checkpoint. Decision (Syed, 2026-11-20): skip it.
- **`coverage_report.py --out-dir runs/coverage_full`**, since the script names its outputs `cp2_coverage_*` and the default dir would overwrite the CP2 figures.

## Interface change requests
None.

## Follow-ups
- The chain does not wait for CP5-C: the CP4 aug model tracks the real ship to < 0.3 px over 20 steps (Results), so an `action_input` change is unlikely. If CP5-C still adds `action_input: adagn+planes`, `final.yaml` needs that key and the run restarts.
- The missed hits (score never increases in rollouts) are the thing to check in CP6's rollout eval of the final checkpoint; more kills in `data/full` and the larger model are the expected fix.
- Human play after the final run: Aunonto and Michael record (≥ 1 h total, games ≥ 236 steps for the T=200 rollout eval), copy to `$S/human/`, `validate_data.py`, then the human-play rollout eval.
- `$S/human/` is free to fill now that the chain has finished.
