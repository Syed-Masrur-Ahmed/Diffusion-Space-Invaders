# cp3-A: Metrics, baselines, visualization, human recorder

## Summary
- `wm/eval/metrics.py`: per-image `psnr`, `ssim`, `lpips` on [0,1] `(N,C,H,W)` tensors, any device, C = 3 or 1 (greyscale is repeated to RGB for LPIPS). PSNR is capped at 100 dB. LPIPS(alex) loads lazily and is cached per device. `METRICS` maps names to functions.
- `wm/eval/baselines.py`: `CopyLastFrame` (the `WorldModelProtocol`; returns a copy of `context[:, -1]`).
- `wm/eval/visualize.py`: `save_grid(path, context, target, pred)`. One row per sample with columns last context | target | prediction | |diff|×4, ×4 nearest upscaling and a 2 px grey gutter, saved as an RGB PNG with imageio.
- `scripts/eval_single_step.py` + `configs/eval/single_step.yaml`: teacher-forced metrics on `n` val windows spread evenly over the val split. Writes `<out_dir>/<model tag>/metrics.json` (mean, std, 95% CI per metric) and `grid.png`. Accepts `--model copy_last` or a checkpoint path.
- `scripts/play_record.py` + `configs/data/human.yaml`: human recorder into `data/human/` at D10 (84/max), `source=human`, seeds from 2,000,000,000, Esc → `end_reason=stopped`.
- **Copy-last baseline on `data/debug` val (2,048 windows):** PSNR 35.45 ± 0.78 dB, SSIM 0.9673 ± 0.0015, LPIPS 0.0105 ± 0.0006.

## Files
Created: `scripts/eval_single_step.py`, `scripts/play_record.py`, `configs/eval/single_step.yaml`, `configs/data/human.yaml`, `tests/test_metrics.py`, `tests/test_play_record.py`, `docs/figures/cp3_copy_last_grid.png`, `progress/cp3-A.md`.
Modified (Phase 0 stubs replaced): `wm/eval/metrics.py`, `wm/eval/baselines.py`, `wm/eval/visualize.py`.

## Tests
`tests/test_metrics.py` (16):
- PSNR: identical images give exactly 100 (the cap); a constant offset of 0.1 gives 20 dB; values are per image; greyscale works; shape mismatch raises.
- SSIM: identical images give 1; noise drops it below 0.9; greyscale works.
- LPIPS (`network`): identical images ≈ 0 (< 1e-4); different images > 0.01; the model is built once per device; greyscale accepted.
- `CopyLastFrame`: returns `context[:, -1]` as a copy, not a view. **On a fake shard** (`encode_rows=True`, rows < 256), its PSNR on every window equals the PSNR between consecutive encoded frames, 10·log10(3·255²) = 52.92 dB.
- `save_grid`: exact layout (size, gutter, ×4 upscaling) and pixel values per column, including the |diff|×4 column computed without uint8 wrap-around; greyscale input saves as RGB.
- `eval_single_step`: `pick_indices` spreads and caps; `evaluate` with copy-last on a fake shard gives the analytic PSNR on all 50 windows; `main` with `--set n=12` writes `metrics.json` and `grid.png`.

`tests/test_play_record.py` (7):
- `next_worker`: the smallest unused worker id ≥ `first_worker`; another person's range doesn't push it up.
- `--hz` outside (0, 15] is rejected (0, −1, 15.5, 30).
- Scripted headless session (`env`, SDL dummy driver, events injected through `get_events`): D held, a Space tap, then A, then Esc. The validator passes. Actions are exactly `RIGHT×3, RIGHTFIRE×3, LEFTFIRE, LEFT×3, -1` (FireLatch keeps fire on for 4 ticks). `source=human`, `end_reason=stopped`, `epsilon=-1`, no terminal row, `seed = HUMAN_SEED_BASE + episode_uid`, attrs 84/max. **Replay** of the seed and the recorded actions reproduces every stored frame (D9).
- Two sessions in one directory get workers 0 and 1 and seeds 2e9 and 2e9 + 1e7, and the dataset validates.

Commands (macOS x86_64, Python 3.12.6, torch 2.2.2, torchmetrics 1.9.0, lpips 0.1.4, ale-py 0.10.2, CPU):
- `pytest -m "not slow and not gpu and not network"` → 186 passed in 57 s (165 at cp2 + 21 new).
- `pytest -m "slow or network or env"` → 24 passed (20 at cp2 + 2 `network` LPIPS + 2 `env` recorder).

## Results
`python scripts/eval_single_step.py --config configs/eval/single_step.yaml --model copy_last --n 2048`
(23,161 val windows in `data/debug/`, K = 4, 2,048 evaluated, CPU, 47 s):

| Model | PSNR (dB) | SSIM | LPIPS (alex) |
|---|---|---|---|
| copy-last | 35.45 ± 0.78 | 0.9673 ± 0.0015 | 0.0105 ± 0.0006 |

(mean ± 95% CI half-width; PSNR std 18.1, SSIM std 0.035, LPIPS std 0.013)

Grid: `docs/figures/cp3_copy_last_grid.png`. The |diff| column isolates exactly what changes between steps: lasers, bullets, the player, and alien animation frames.

## Deviations
- **Which val windows.** "`--n 2048` on the val split" doesn't say which windows. `pick_indices` takes `n` indices spread evenly over the whole val split (`round(linspace(0, len-1, n))`). This is deterministic, needs no RNG, and covers every val episode. The val split is in order, so the first 2,048 windows would come from only a few episodes.
- **Config plus flags.** The roadmap gives `--model` and `--n`; the script also takes `--config`/`--set` (4.12). The flags override `model` and `n` in the config. `metrics` in the config selects which metrics run (the tests skip LPIPS so the fast suite needs no download).
- **Metric API.** Each metric returns per-image values `(N,)`, not a batch mean, so callers can compute CIs (CP4-A needs them). The 95% CI is the normal approximation 1.96·std/√n.
- **SSIM settings.** torchmetrics defaults (Gaussian 11×11, σ = 1.5, k1 = 0.01, k2 = 0.03), `data_range=1`.
- **`metrics.json` adds `psnr.capped_fraction`**, the share of windows at the 100 dB cap (identical frames). See Follow-ups.
- **Recorder worker ids.** A session writes as the smallest unused worker id ≥ `first_worker` (config), so repeated sessions in one directory never clash. Episode seed = `HUMAN_SEED_BASE + worker·1e7 + episode` = `HUMAN_SEED_BASE + episode_uid`, matching the collector's D18 rule.
- **Recorder controls.** Keys are tracked from KEYDOWN/KEYUP events (cleared on focus loss) instead of `pygame.key.get_pressed()`, so a test can inject a scripted key sequence through `run(get_events=...)`. A new `make_env(seed)` is made per episode so each episode's seed is explicit. After a game over, R starts a new episode; Esc or closing the window ends any open episode as `stopped` and quits. Pressing Esc before the first step stores a valid 1-row episode (0 windows).

## Interface change requests
None.

## Follow-ups
- **CP3 integration check 2 needs a checkpoint loader.** For a checkpoint, `eval_single_step.py` calls `WorldModel.from_checkpoint(path, device=...)` (the frozen 4.10 API) and reads `K`, `C`, `H`, `W` from it. `WorldModel` is a CP4-C stub, and the checkpoint layout is CP3-B's. To run the check at CP3, the integrator needs a minimal `WorldModel` (UNet from the checkpoint config → `EDMDenoiser` → `sample` with EMA weights). It can be built from `progress/cp3-B.md`, either as an early part of CP4-C or as a small CP3 integration commit. Nothing in this script needs to change.
- **PSNR mean and the cap.** 6.8% of copy-last windows are identical frames at 100 dB, which pulls the mean up (median 31.06 dB, mean over non-identical windows 30.72 dB). Compare models on the same windows (same `n`, `val_fraction`, dataset), and use LPIPS too, as 4.11 says.
- **LPIPS needs AlexNet weights** (233 MB, `~/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`) on first use. Discovery compute nodes may have no internet; download once on a login node (`python -c "import lpips; lpips.LPIPS(net='alex')"`) before submitting eval jobs.
- **Human data.** `play_record.py` has not been played by a person yet, only in the scripted headless test. The open question in 1.4 (how much to record) still stands. When recording on several machines into datasets that will be merged, give each person their own `first_worker` (e.g. Aunonto 0, Syed 100, Michael 200).
- `data/human/` is not in git, like the other datasets.
