# cp4-B: Context noise augmentation runs

## Summary
- `wm/train/trainer.py`: `context_noise(context, n_levels, aug_max, mode)` implements 4.8.
  - Draws `level ~ U{0..n_levels-1}`, sets `σ_aug = level/(n_levels-1)·aug_max`, and returns `context + σ_aug·ε` (in [-1,1] units, **not clamped**) with `ctx_level` (B, K) long.
  - Modes: `shared` (one level per sample, repeated over K) and `per_frame` (one level per frame).
  - `aug_max = 0` returns the context unchanged with zero levels and draws **no** random numbers, so the aug-off run follows exactly the CP3 trainer's RNG stream.
- `Trainer._inputs(batch, augment)` (renamed from `drop`) applies action dropout and context noise to training batches only. The two are independent draws. Val loss and EMA sample grids use clean context with `ctx_level = 0`.
- Logging, every `log_every` steps: TensorBoard histogram `train/ctx_level` (bins centred on 0..n_levels-1) and scalars `loss_by_ctx_level/<l>` (no extra forward passes), on top of the existing loss and loss-by-σ scalars.
- `configs/train/debug_noaug.yaml` and `configs/train/debug_aug.yaml`: full copies of `debug.yaml` that differ only in `name` and `train.aug_max` (0 vs 0.7, both `shared`).
- Both 50k-step Discovery runs finished on 2026-11-17 (`runs/20261117-0431_debug_noaug`, `runs/20261117-0431_debug_aug`). The horizon comparison runs at CP4 integration.

## Files
- Modified: `wm/train/trainer.py` (`context_noise`, `_inputs`, `train_step` returns `ctx_level`, `_log_ctx_levels`, config validation, `n_levels` check).
- Created: `tests/test_noise_aug.py`, `configs/train/debug_noaug.yaml`, `configs/train/debug_aug.yaml`, `progress/cp4-B.md`.
- Not touched: `wm/diffusion/*`, `wm/models/*`, `wm/inference.py`, `wm/data/dataset.py`, `tests/test_trainer.py`, `configs/train/debug.yaml`.

## Tests
`tests/test_noise_aug.py`, 19 tests (CPU, `TinyDenoiserNet`, `make_fake_shard(encode_rows=True)` through the trainer):
- **`aug_max = 0`:** the context is returned bit-identical, `ctx_level` is all-zero long (B, K), and the torch RNG state is unchanged (nothing drawn).
- **Noise std per level** (shared and per_frame, 2,000×4×3×8×8): the std of `ctx_aug - ctx` is within 5% of `σ_aug` for every level 1..9, and the mean is ≈ 0.
- **Level 0** adds exactly zero noise with `aug_max = 0.7`.
- **Not clamped:** an all-ones context goes above 1.5.
- **Modes:** `shared` gives one level across all K frames. `per_frame` levels vary within > 95% of samples (expected 99.9%).
- **Uniform levels:** each of 0..9 is within 10% of the expected count over 20,000 draws (≥ 5 binomial std), in both modes.
- **Net spy:** `context_noise` is wrapped and the net's forward is hooked during a trainer step, in both modes.
  - Its input is the loader's clean context.
  - The net receives exactly the returned `ctx_level` (long, (B, K)) and the returned noisy context, flattened.
  - Per frame, the context differs from the clean one iff `ctx_level > 0`.
- **Clean eval paths:** with `aug_max = 0.7`, the val loss and the EMA sample grid feed the net `ctx_level = 0` and the unmodified grid context.
- **Logging:** a 5-step run writes the `train/ctx_level` histogram and `loss_by_ctx_level/*` scalars.
- **Exact resume with `aug_max = 0.7`** (shared and per_frame, fp32, `num_workers = 0`): 10 steps, save, resume to 20 equals 20 straight steps (`assert_close`, rtol 1e-6) for `model` and `ema`. The same run with `aug_max = 0` gives different weights, so the noise is really active. ~1 s each, so not marked `slow`.
- **Configs:** the resolved `debug_noaug`/`debug_aug` configs have the same keys as `debug.yaml` and differ only in `name` and `train.aug_max`; `ctx_noise_mode = shared`.
- **Invalid config:** `ctx_noise_mode` other than shared/per_frame, or `aug_max < 0`, raises `ValueError`.
- **Mutation check:** each of these breaks at least one test: passing zero `ctx_level` with a noisy context; noising val/grid batches; dividing by `n_levels` instead of `n_levels-1`; per-frame draws in shared mode; never drawing level 0.

Commands (macOS arm64, M3, Python 3.13.2, CPU):
- `pytest -m "not slow and not gpu and not network"` → 262 passed, 8 deselected in 40 s (243 on `main` + 19 new)
- `pytest -m "slow or network or env"` → 25 passed in 79 s (includes the slow `num_workers=2` trainer resume test); `pytest tests/test_trainer.py -k resume` (incl. slow) → 4 passed; `pytest -m gpu` → 1 skipped (no CUDA)
- The CP3 trainer tests (`tests/test_trainer.py`) pass unchanged, including the slow `num_workers=2` resume test.

## Results
- **Dry runs on `data/debug/`** with the real `small` U-Net (14,263,683 parameters), CPU:
  - `python scripts/train.py --config configs/train/debug_aug.yaml --dry-run` → `dry run: 5 steps, last loss 2.4462, val loss 2.42677, samples (8, 3, 84, 84)` in 275 s.
  - `... debug_noaug.yaml --dry-run` → `last loss 2.3350, val loss 2.42676, samples (8, 3, 84, 84)` in 267 s.
  - Val losses agree to 2e-6 (clean context in both; 5 steps barely move the weights). Train losses differ because the aug run draws extra random numbers (levels, ε) before σ.
- **σ_aug per level** (`aug_max = 0.7`, `n_levels = 10`; ×127.5 for uint8 units):

  | level | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | σ_aug | 0 | 0.0778 | 0.1556 | 0.2333 | 0.3111 | 0.3889 | 0.4667 | 0.5444 | 0.6222 | 0.7 |
  | uint8 | 0 | 9.9 | 19.8 | 29.8 | 39.7 | 49.6 | 59.5 | 69.4 | 79.3 | 89.3 |

- **Discovery runs** (2026-11-17, `gpuq`, one NVIDIA L40S each on node `l40sx8`, both started 04:31):

  | | `debug_noaug` | `debug_aug` |
  |---|---|---|
  | config | `configs/train/debug_noaug.yaml` (`aug_max` 0) | `configs/train/debug_aug.yaml` (`aug_max` 0.7, shared) |
  | Slurm job | 11093964 | 11093965 |
  | run dir | `runs/20261117-0431_debug_noaug` | `runs/20261117-0431_debug_aug` |
  | final EMA checkpoint | `runs/20261117-0431_debug_noaug/checkpoints/ckpt_last.pt` (`ema`) | `runs/20261117-0431_debug_aug/checkpoints/ckpt_last.pt` (`ema`) |
  | steps reached | 50,000 (`finished at step 50000`, one job, no resume) | 50,000 (same) |
  | wall time | 2 h 22 min (last checkpoint 06:53) | 2 h 25 min (06:56) |
  | throughput (median logged) | 387 samples/s | 383 samples/s |
  | train loss at step 50,000 (last 50 steps) | 0.0010 | 0.0018 |
  | grad norm at step 50,000 | 0.053 | 0.277 |
  | val loss (clean context), steps 48k / 49k / 50k | 0.0015 / 0.0012 / 0.0015 | 0.0014 / 0.0014 / 0.0020 |
  | final sample grid | `runs/20261117-0431_debug_noaug/samples/step_0050000.png` | `runs/20261117-0431_debug_aug/samples/step_0050000.png` |
  | level histogram | n/a (all 0) | TensorBoard `train/ctx_level` (not yet inspected) |

  - Run dirs and checkpoints are on Discovery under the repo's `runs/`. Data: `--set data.root=/dartfs-hpc/scratch/f0079km/wm/data/`.
  - Throughput is ~1.3× the CP3 run (292 samples/s), so a 50k run takes ~2.4 h on an L40S.
  - **Train loss is not comparable across the runs:** the aug run's train loss includes samples with noisy context (levels 1-9), which are harder. Its higher grad norm fits that.
  - **Val loss (clean context) is close:** both are in 0.0012-0.0020 over the last 3k steps. The step-to-step variation (noaug 0.0012 → 0.0015, aug 0.0014 → 0.0020) is about as large as the gap, so one-step val loss does not separate the runs. The aug model spends some capacity on noisy contexts, so a slightly higher clean-context loss is expected. The test that matters is the rollout horizon curve.
  - `aug_max`, mode and seed are taken from each job's config line. Checking them from the checkpoints (`grep -A3 aug_max runs/20261117-0431_debug_*/config.yaml`) was not done; the first try ran without the venv.

## Training on Discovery (for Syed)
Both runs start fresh from step 0 with `seed: 0` on the same data. Don't reuse `runs/20261105-0558_debug_small`.
The CP3 small run did ~292 samples/s, so 50k × 64 ≈ 3 h per run. The two runs are independent and can run at the same time.
The names `debug_noaug` and `debug_aug` differ, so `--resume auto` never picks up the other run (or the old `debug` run).

```bash
# Login node, repo root, on this branch (or main after integration):
source .venv/bin/activate
mkdir -p runs
SCRATCH=<scratch>/data                    # holds debug/ (rsync -a data/debug $SCRATCH/)
python scripts/train.py --config configs/train/debug_aug.yaml   --dry-run --set data.root=$SCRATCH
python scripts/train.py --config configs/train/debug_noaug.yaml --dry-run --set data.root=$SCRATCH

# Each run: one job, plus afterany follow-ups in case the time limit is under ~3.5 h.
# Each follow-up auto-resumes ckpt_last.pt and exits at once if the run already reached 50k.
for c in debug_noaug debug_aug; do
  jid=$(CONFIG=configs/train/$c.yaml sbatch --parsable --partition=gpuq --time=<HH:MM:SS> \
        scripts/slurm/train.sbatch --set data.root=$SCRATCH)
  jid=$(CONFIG=configs/train/$c.yaml sbatch --parsable --dependency=afterany:$jid --partition=gpuq \
        --time=<HH:MM:SS> scripts/slurm/train.sbatch --set data.root=$SCRATCH)
  CONFIG=configs/train/$c.yaml sbatch --dependency=afterany:$jid --partition=gpuq --time=<HH:MM:SS> \
        scripts/slurm/train.sbatch --set data.root=$SCRATCH
done
```

- Stop a job only with `scancel --signal=USR1 --batch <jobid>`. A plain `scancel` can lose up to `save_every` (5,000) steps.
- Check `runs/slurm-<jid>.out` for `new run runs/<stamp>_debug_<x>`, then `resumed ... at step N` in follow-ups and `finished at step 50000`.
- TensorBoard: `tensorboard --logdir_spec noaug:runs/<stamp>_debug_noaug/tb,aug:runs/<stamp>_debug_aug/tb`. Compare `val/loss` (clean context in both), and look at `train/ctx_level` (histograms tab) and `loss_by_ctx_level/*` for the aug run.
- Hand-off for the horizon comparison (CP4-A's `eval_rollouts.py`, at integration): the two `ckpt_last.pt` files above, EMA weights, with inference `ctx_level` swept over {0, 1, 2}.

## Deviations
- **Branched from `main` (9856e3c), not tag `cp3`.** `cp3` includes the `cp3/loader` merge but not the Milestone 1 sampler fix (D21, 932466e).
- **Val loss and sample grids use clean context** (`ctx_level = 0`, no noise), so the two runs' val curves are comparable. 4.8 only says inference uses `ctx_level` default 0; it does not say how to validate.
- **`Trainer._inputs(batch, drop)` → `_inputs(batch, augment)`**, since the flag now controls both action dropout and context noise. It is private, and all callers are in `trainer.py`.
- **Single source for `n_levels`:** `model.n_levels` (passed to `UNetConfig.n_levels`). There is no `train` copy. `Trainer.__init__` asserts that every `level_emb` embedding in the net (UNet's `emb.level_emb`, TinyDenoiserNet's `level_emb`) has `model.n_levels` rows.
- **RNG:** levels come from the global CPU torch RNG (like `drop_actions`), and ε from the global RNG of the context's device (`randn_like`). Both are in `rng_states` on CPU and CUDA, so exact resume (D19) holds. On MPS the MPS generator is not saved; MPS is for local debugging only.
- **Per-level loss** is keyed by the level of the newest context frame. That is the sample's level in `shared` mode and an approximation in `per_frame` mode.
- **The configs are full copies, not inheritance:** `load_config` has no include mechanism. A test checks that the resolved configs differ from `debug.yaml` only in `name` and `train.aug_max`. Resolved diff `debug_noaug` vs `debug_aug`: `name: debug_noaug → debug_aug`, `train.aug_max: 0.0 → 0.7`, nothing else.
- **The resume test with aug is not marked `slow`** (~1 s with `num_workers = 0`). The existing `num_workers = 2` resume test in `test_trainer.py` still covers workers.

## Interface change requests
None.

## Follow-ups
- Look at the aug run's `train/ctx_level` histogram (should be flat over 0-9) and `loss_by_ctx_level/*` in TensorBoard, and compare the two `step_0050000.png` grids.
- Record the `gpuq` limits for ROADMAP 1.4: 5-day time limit; GPUs `nvidia_l40s` (8 per node) and `nvidia_a100_80gb_pcie_3g.40gb` MIG slices (4).
- Integration (CP4-A's `eval_rollouts.py`): horizon curves for `debug_noaug` vs `debug_aug` at inference `ctx_level` ∈ {0, 1, 2}, plus copy-last.
- CP6-B: the demo spec doesn't mention `ctx_level`, so it would use `predict_next`'s default of 0. Pass the level CP4 integration picks (from the demo config), and optionally add a key to change it live.
- Still open from CP3: the HDF5 worker close hook in `wm/data/dataset.py` (out of scope here).
- CP6-C ablation: `per_frame` mode is implemented and tested, but no run uses it yet.
