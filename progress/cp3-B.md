# cp3-B: Trainer

## Summary
- `wm/train/trainer.py`: `RunConfig` (nested dataclasses loaded with `wm.config`), `Trainer`, and `run`.
  - AdamW, linear warmup then constant or cosine, grad clip 1.0, optional gradient accumulation, and optional `torch.compile`.
  - bf16 autocast when CUDA supports it, fp32 otherwise.
  - EMA (0.999). Action dropout (whole sequence → `NULL_ACTION`). `ctx_level = zeros(B, K)`.
  - TensorBoard: loss, loss by σ bucket, lr, grad norm, samples/s, and val loss.
  - Sample grids from the EMA model.
  - Run dirs, atomic `ckpt_last.pt`, `ckpt_<step>.pt` pruned to `keep_last`.
  - **Exact resume** (RNG states plus data position), `--resume RUN_DIR|auto`, and save-and-exit on SIGUSR1/SIGTERM.
- `scripts/train.py --config ... [--set ...] [--resume RUN_DIR|auto] [--dry-run]`.
- `scripts/slurm/train.sbatch` (1 GPU, `--signal=B:USR1@300`, `exec python scripts/train.py --config $CONFIG --resume auto "$@"`) and `scripts/slurm/eval.sbatch` (runs `$CMD`).
- `configs/train/debug.yaml`: K=4, batch 64, 50k steps, the small U-Net (`arch: unet`), and `data/debug/`. Also `model.arch: tiny`, which builds `TinyDenoiserNet` for tests and the Discovery smoke job.
- Not done yet: the 1,000-step Discovery job. Commands are below, for Syed to submit.

## Files
- Created:
  - `configs/train/debug.yaml`
  - `scripts/train.py`
  - `scripts/slurm/train.sbatch`, `scripts/slurm/eval.sbatch`
  - `tests/test_trainer.py`
  - `progress/cp3-B.md`
- Modified: `wm/train/trainer.py` (Phase 0 stub replaced).
- Not touched: `tests/fixtures.py` (no new fixtures needed), `wm/data/dataset.py`, `wm/config.py`.

## Tests
`tests/test_trainer.py`, 12 tests, on a fake shard (`encode_rows`, 6×8 non-square frames, episodes [12, 3, 20, 9, 15, 2, 11] rows) with `model.arch=tiny`, on CPU in fp32:
- **Overfitting one fixed batch** for 300 steps lowers the loss by ≥ 5×. The loss is measured with fixed σ/ε draws (`eval_loss`).
- **Resume equals a straight run:** 10 steps, save, resume to 20, then `model` and `ema` match 20 straight steps (`assert_close`, rtol 1e-6).
  - Batch size 3 makes the resume land mid-epoch and the run cross an epoch; the test asserts both.
  - Run with `num_workers=0` (fast) and `num_workers=2` (marked `slow`, see Results).
  - A mutation check confirmed the test fails if the batch skip is dropped, the RNG restore is dropped, or the data position is reset.
- **Action dropout:** the empirical rate over 20k samples is 0.1 ± 0.03. Dropped rows are all `NULL_ACTION`, other rows are unchanged, and p=0 is the identity.
- **Checkpoint:** keys are exactly the 4.12 set, `step` is correct, and `rng_states` holds python, numpy and torch.
  - The run dir has `config.yaml`, `git_commit.txt`, `tb/`, `checkpoints/` and `samples/` (with a PNG).
  - Pruning keeps the last 3 `ckpt_<step>.pt` files plus `ckpt_last.pt`.
- **`--dry-run`** (subprocess) runs 5 steps, prints `dry run: 5 steps`, and writes no run dir.
- **SIGUSR1** (subprocess): after ≥ 3 logged steps, SIGUSR1 leads to `ckpt_last.pt`, the `caught SIGUSR1` message, and exit 0. `--resume auto` with `train.steps=step+3` then prints `resumed <run_dir> at step <step>`, ends at step+3 in the same run dir, and creates no new dir. Takes about 3 s.
- **Alignment through the trainer's loader:** decoded target row `i`, context rows `i-K..i-1`, and actions equal to the shard's `actions[i-K..i-1]`. The net receives context `(B, K·C, H, W)`, `ctx_level` all-zero long `(B, K)`, and the batch actions.
- **EMA:** it equals the weights exactly during warmup and diverges after.
- **Smaller checks:**
  - `EpochBatchSampler`: deterministic per epoch, differs across epochs, skip yields the tail of the same permutation, drop_last.
  - `find_auto_resume`: newest stamp with a `ckpt_last.pt`, exact name match (`debug` ≠ `debug_aug` ≠ `x_debug`).
  - The `debug.yaml` defaults.

Commands (macOS arm64, 10 CPUs, Python 3.13.2, torch 2.14.1, CPU only):
- `pytest -m "not slow and not gpu and not network"` → 176 passed, 5 deselected, in 26 s (165 at cp2).
- `pytest tests/test_trainer.py` (including slow) → 12 passed in 69 s. The `num_workers=2` resume test alone takes ~65 s.
- This workflow has no `gpu` or `network` tests. The CP3-B spec has no GPU timing test; that is CP3-C's.

## Results
- **Exact resume holds with `num_workers > 0`** (the `num_workers=2` test passes).
- **Real data, `data/debug/`**, tiny net (`tiny_hidden=32`, 16,779 parameters), CPU, lr 2e-3, 6 workers, 300 steps:
  - Loss 1.15 → 0.19, val loss (512 fixed windows) 0.39 at step 150 and 0.16 at step 300.
  - 150–190 samples/s on the laptop CPU.
  - The sample grid (last context | target | prediction) is RGB, square, and the right way round.
- **`train.sbatch` run locally under `bash`**: SIGUSR1 sent to the shell PID reached Python through `exec`. It saved at step 24 and exited 0, and a second invocation printed `resumed ... at step 24` and finished at step 40.
- **`--dry-run` on `debug.yaml`:**
  - With `--set model.arch=tiny`, it passes: 5 steps, a val loss, and samples (8,3,84,84).
  - With `arch: unet`, it raises `NotImplementedError: CP3-C` until the U-Net merges.
- **Shutdown cost (not a trainer bug):** a persistent DataLoader worker that has an HDF5 file open does not exit on its own at shutdown. Torch waits its 5 s join timeout per worker and then terminates it.
  - I reproduced it with CP2's own `make_dataloader`, which also takes 5–10 s to tear down.
  - It does not happen with `in_memory=True`, a tensor dataset, or non-persistent workers.
  - Effect: process exit takes ~5 s per worker, 40 s with 6 train and 2 val workers. That's after the checkpoint is saved, and well inside the 300 s signal margin.

## Deviations
- **The train loader is built in the trainer, not with `make_dataloader`.** It uses the same `WindowDataset` and the same settings (persistent workers, `pin_memory` on CUDA, `drop_last`, `seed + worker_id` worker seeding, lazy HDF5).
  - The difference: it takes an `EpochBatchSampler`, whose epoch `e` is `randperm(n, Generator(seed + e))`, and a dedicated loader `generator`.
  - Why: torch draws each new iterator's base seed from `loader.generator`. In `make_dataloader` that generator is also the shuffle generator. A resumed run creates its persistent iterator at a different point than a straight run, so with workers > 0 the shuffle order would change. With no generator, the draw would come from the global torch RNG and shift σ/ε.
  - **Data position on resume:** `consumed = step × grad_accum` batches, so epoch = `consumed // batches_per_epoch` and skip = `consumed % batches_per_epoch`. The sampler skips without loading anything.
  - This assumes the data, `batch_size` and `grad_accum` are unchanged across a resume. The position is derived from `step`, so the checkpoint needs no extra key.
- **Val loss uses a fixed, evenly spaced subset** of `train.val_windows` (2048) val windows. It is computed under `torch.random.fork_rng` with fixed seeds, so it is comparable across steps and never touches the training RNG. Its loader also builds its own DataLoader (≤ 2 persistent workers) over `Subset(val_set, ...)`.
- **The worker init is the trainer's own** (`_worker_init`). It seeds like CP2's `_init_worker`, closes inherited handles through a `Subset`, and ignores SIGUSR1 in workers.
  - SIGTERM keeps its default in workers. If workers ignored it, torch's `terminate()` fallback could not stop the hung workers above, and they would be left behind.
  - Consequence: a SIGTERM sent to every process in the job (e.g. `scancel` with no `--signal`) can kill the workers before the main process saves. torch then raises in the main process, and the run loses up to `save_every` steps, which 4.15 allows.
  - The `B:USR1@300` path signals only the batch shell (= Python) and always saves. A SIGTERM sent only to the Python process also saves.
- **Sample grids use a private helper** (`_save_grid` in `trainer.py`): one row per sample, last context | target | prediction, ×4 nearest, saved with `imageio`. This is because `wm/eval/visualize.py` (CP3-A) is not on this branch. A later checkpoint can switch to `save_grid`. The grid windows are fixed, evenly spaced val windows, sampled with the EMA model in eval mode under `inference_mode`, with the default sampler and `Generator(seed)`. The EMA model's previous mode is restored afterwards.
- **`model.arch: tiny`** builds `tests/fixtures.py::TinyDenoiserNet` (lazy import; the repo root is added to `sys.path`). All tests use it, and so can the Discovery smoke job before the U-Net merges.
- **U-Net fields live in `debug.yaml`** (`model.ch: [64,128,256]`, `n_res`, `emb_dim`, `n_levels`, `attn_lowest`, `dropout`), not loaded from CP3-C's `configs/model/small.yaml`. This keeps the merge clean: nothing here depends on a file CP3-C is writing. `C` and `H` come from the data, and `K` from `data.K`. If CP3-C's "small" values differ, the integrator updates `debug.yaml`.
- **Checkpoint contents:**
  - `model` and `ema` are the **network's** state dicts (UNet keys, with no `net.` prefix and no `_orig_mod.` from `torch.compile`).
  - `config` is `dataclasses.asdict(RunConfig)`.
  - `rng_states` is `{python, numpy, torch, cuda}`.
  - Checkpoints are loaded with `map_location="cpu"`, so the RNG tensors stay on CPU.
  - `ckpt_<step>.pt` is a copy of `ckpt_last.pt`, made through a temp file and `os.replace`.
- **Resume uses the command-line config**, not the one stored in the checkpoint. This lets `--set train.steps=...` extend a run. `config.yaml` is rewritten at each start.
- **EMA:** `ema ← weights` for steps `≤ warmup_steps`, then `ema.lerp_(w, 1 - decay)` every step. Buffers are copied.
- **LR:** `(s+1)/warmup` during warmup, then 1 (constant), or cosine to 0 at `train.steps` (`lr_schedule: cosine`).
- **Gradient accumulation:** one optimizer step consumes `grad_accum` batches. Each batch's loss is divided by `grad_accum`.
- **Dry run:** 5 steps, one val pass, one sample batch; it writes nothing (no run dir).
- **Run dir names:** if `runs/<stamp>_<name>` already exists (two starts in the same minute), the new dir is `<stamp>-1_<name>`. `--resume auto` orders by (stamp, n).
- **Device:** cuda, then mps only with `train.allow_mps: true`, then cpu. On MPS, the val-loss `fork_rng` does not fork the MPS generator (CPU and CUDA only). MPS is for local debugging only.
- **Added config keys** that 4.12 does not fix:
  - `name`, `seed`
  - `data.root`, `data.dirs`
  - `train.lr_schedule`, `train.allow_mps`, `train.runs_dir`, `train.log_every`, `train.val_windows`, `train.n_samples`
  - `model.tiny_hidden`
- `aug_max` (default 0) and `ctx_noise_mode` (default `shared`) exist but are not used yet (CP4-B).

## Interface change requests
None. Suggestion for whoever next owns `wm/data/dataset.py`: let `make_dataloader` accept a `batch_sampler` and a separate base-seed generator, so the trainer can use it directly.

## Follow-ups
**Discovery done criterion (for Syed).** Run from the repo root on Discovery, after `setup.sh` and the CUDA PyTorch install. Fill in the partition and time from section 1.4. The U-Net isn't merged, so use `model.arch=tiny` and a separate run name, so that a later real `debug` run's `--resume auto` doesn't pick up the tiny run.

```bash
rsync -a data/debug <scratch>/data/            # or regenerate with scripts/collect.py (D18)
mkdir -p runs                                   # Slurm needs it for runs/slurm-%j.out
source .venv/bin/activate
python scripts/train.py --config configs/train/debug.yaml --dry-run \
    --set data.root=<scratch>/data --set model.arch=tiny          # quick check on the login node

ARGS="--set data.root=<scratch>/data --set model.arch=tiny --set name=debug_tiny"
# Job 1: 1,000 steps from scratch.
jid=$(CONFIG=configs/train/debug.yaml sbatch --parsable --partition=<gpu-partition> --time=00:30:00 \
      scripts/slurm/train.sbatch $ARGS --set train.steps=1000)
# Job 2: starts after job 1 ends, auto-resumes the same run dir, continues to 2,000.
CONFIG=configs/train/debug.yaml sbatch --dependency=afterany:$jid --partition=<gpu-partition> --time=00:30:00 \
      scripts/slurm/train.sbatch $ARGS --set train.steps=2000
```

To check:
- `runs/slurm-<job1>.out` shows `new run runs/<stamp>_debug_tiny` and `finished at step 1000`.
- `runs/slurm-<job2>.out` shows `resumed runs/<stamp>_debug_tiny at step 1000` and `finished at step 2000`.
- To test the signal path on the cluster, run `scancel --signal=USR1 --batch <jobid>` on a running job. It should print `caught SIGUSR1: saved ckpt_last.pt at step N, exiting`.
- TensorBoard: `tensorboard --logdir runs/<stamp>_debug_tiny/tb`.

Once CP3-C's U-Net is merged, the real CP3 run is the same command without `ARGS`' `model.arch` and `name` overrides, chained with `afterany` until 50k steps. Data loading: use the lazy loader with the 6 default workers; set `--cpus-per-task` ≥ 8.

**Other follow-ups:**
- **Integrator (CP3):** check that `debug.yaml`'s `model.ch/n_res/emb_dim` match CP3-C's `configs/model/small.yaml`, and that `UNet(UNetConfig(C, K, H, ch, n_res, emb_dim, n_levels, attn_lowest, dropout))` builds. Then run `--dry-run` with `arch: unet`.
- The slow HDF5 worker shutdown: closing the shard handles when a worker shuts down would remove it. That needs a hook in `wm/data/dataset.py`, which this workflow doesn't own. Low priority (it costs time only at process exit).
- **CP4-B:** context noise goes in `Trainer._inputs` (currently `torch.zeros_like(actions)`), drawn from the global torch RNG so exact resume keeps working. Log the level distribution there.
- **CP4-C:** `WorldModel.from_checkpoint` can load `ckpt["ema"]` straight into `UNet(UNetConfig(...))`, building the config from `ckpt["config"]["model"]` plus `data.K`, with `C`/`H` from the data (`data.grayscale`, resolution 84).
- `git_commit.txt` records `HEAD` only and doesn't record uncommitted changes.
