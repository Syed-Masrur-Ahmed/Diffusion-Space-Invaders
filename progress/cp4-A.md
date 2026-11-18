# cp4-A: Rollouts and horizon evaluation

## Summary
- `wm/eval/rollout.py`: `rollout(model, frames0, actions, T, **sample_kwargs)` with the 4.11 alignment (g_j from the last K frames and a_{j-K}..a_{j-1}; output t is g_{K+t}, horizon h = t+1), `horizon_metrics` (per-frame `(B,T)` values for any `METRICS` name), and `load_trajectories` (n episodes spread evenly over the eval set, rows `start..start+K+T-1`, uint8 as stored).
- `wm/eval/plots.py`: `plot_horizon` (one panel per metric, one line per model, 95% CI band, log x) and `save_rollout_video` (real | generated | |diff|×4 MP4, ×4 nearest upscaling, optional display-only widening).
- `scripts/eval_rollouts.py` + `configs/eval/rollouts.yaml`: `--models copy_last <ckpt>...`, `--T`, `--n`, `--set ctx_levels=[0,1,2]` (each checkpoint runs once per inference context level, for the CP4 integration check). Writes `metrics.json`, `metrics.csv` (h = 1..T), `horizon.png` and `videos/<model>_traj<i>.mp4` to `out_dir`.
- **Copy-last horizon curve on `data/eval`** (64 trajectories, T = 200): PSNR 30.04 dB at h=1, falling to 15.58 at h=200. LPIPS goes from 0.0085 to 0.242. Figure: `docs/figures/cp4_copy_last_horizon.png`.
- The CP3 model's curve is **not run yet**: the checkpoint isn't on this laptop, and a CPU run would take hours here. The command for Discovery is below.

## Files
Created: `scripts/eval_rollouts.py`, `configs/eval/rollouts.yaml`, `tests/test_rollout.py`, `docs/figures/cp4_copy_last_horizon.png`, `progress/cp4-A.md`.
Modified (Phase 0 stubs replaced): `wm/eval/rollout.py`, `wm/eval/plots.py`.

## Tests
`tests/test_rollout.py` (13, all in the fast suite; fake shards at 8×8 with `encode_rows=True`, or 84×84 random for the script):
- `load_trajectories`: every frame decodes to row `episode start + j` and the actions equal the stored `actions[start..start+K+T-2]`. Episodes that are too short are skipped. The picks spread evenly. Asking for more trajectories than there are long-enough episodes raises. The `start` offset shifts frames and actions together.
- **Roadmap tests:** with `CopyLastFrame`, every generated frame equals the last context frame. **Index-alignment fixture:** a model that outputs "last frame's encoded index + 1" produces g_j that decodes to row j for every j. A model that writes `actions[:, -1]` into the pixels shows that frame j encodes a_{j-1}. A cheating oracle (it looks up f_{idx+1}) reproduces the real frames exactly, with PSNR 100 and SSIM 1 (±1e-3) at every horizon.
- `horizon_metrics` is per frame (one 20 dB frame among 100 dB ones). `rollout` passes `num_steps`/`cfg_scale`/`ctx_level` to every `predict_next` call and rejects wrong `frames0`/`actions` shapes. After the first step the context holds only generated frames.
- `plot_horizon` writes a PNG. `save_rollout_video` writes an MP4 of the exact size (`H·4 × 3·W·4 + 2 gutters`) and frame count.
- Script end to end: copy_last plus a `TinyDenoiserNet` checkpoint at ctx levels {0, 2}. It checks the JSON (per-horizon and report entries, ctx level recorded, levels give different metrics), the CSV (3 models × T rows), the figure and the videos. The same command run twice gives identical metrics.

Commands (macOS x86_64, Python 3.12.6, torch 2.2.2, CPU):
- `pytest -m "not slow and not gpu and not network"` → 257 passed, 1 skipped (needs CUDA/MPS) in 117 s (244 at `milestone1` + 13 new).
- `pytest -m "slow or network or env"` on `milestone1` before this work: 25 passed. This workflow adds no marked tests.

## Results
`python scripts/eval_rollouts.py --config configs/eval/rollouts.yaml --models copy_last --T 200 --n 64`
(`data/eval/`, 64 of 100 episodes, `start` 32, K = 4, CPU, 163 s). Mean ± 95% CI over trajectories:

| h | 1 | 2 | 5 | 10 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|---|---|
| PSNR (dB) | 30.04 ± 1.00 | 27.27 ± 1.00 | 23.05 ± 0.80 | 20.45 ± 0.16 | 18.40 ± 0.17 | 15.42 ± 0.07 | 16.05 ± 0.13 | 15.58 ± 0.30 |
| SSIM | 0.966 ± 0.009 | 0.943 ± 0.012 | 0.887 ± 0.012 | 0.833 ± 0.006 | 0.761 ± 0.007 | 0.612 ± 0.004 | 0.628 ± 0.006 | 0.558 ± 0.022 |
| LPIPS | 0.0085 ± 0.0020 | 0.0140 ± 0.0027 | 0.0262 ± 0.0025 | 0.0358 ± 0.0014 | 0.0601 ± 0.0031 | 0.0860 ± 0.0023 | 0.1570 ± 0.0084 | 0.2424 ± 0.0094 |

- Figure: `docs/figures/cp4_copy_last_horizon.png`. Videos: `runs/eval_rollouts/cp4_copy_last/videos/` (not in git).
- PSNR bottoms out at h ≈ 64 (14.94 dB) and rises to about 18 dB near h ≈ 128 before falling again. SSIM shows the same bump. This is most likely the alien formation sweeping back to near its starting position: all trajectories start at the same game phase, so the bump lines up across them. LPIPS keeps rising. It is the better long-horizon metric, as 4.11 says.
- copy-last at h = 1 here (30.04 dB) is below its single-step score on debug val (35.45 dB). That number is pulled up by identical frames at the 100 dB cap, and these trajectories start after the start-up phase, where every frame changes.

**CP3 model curve, for Syed (Discovery).** The integration check (4.11, CP4 item 1) needs the CP3 `small` checkpoint and the two CP4-B checkpoints. Once CP4-B's runs finish, copy-last and both of its models can share one command:

```bash
CMD="python scripts/eval_rollouts.py --config configs/eval/rollouts.yaml \
  --models copy_last runs/20261105-0558_debug_small/checkpoints/ckpt_last.pt \
  --T 200 --n 64 --set ctx_levels=[0,1,2] --set out_dir=runs/eval_rollouts/cp4_cp3model" \
  sbatch --partition=gpuq --time=01:00:00 scripts/slurm/eval.sbatch
```

For the integration figure, add the aug-off and aug-on checkpoints after `--models` and save `horizon.png` as `docs/figures/cp4_horizon.png`. Cost: n · T · num_steps forward passes at batch `batch_size` (16) per model and level. That's 64 · 200 · 3 = 38,400 images in 2,400 batch-16 calls, a few minutes per entry on a GPU. LPIPS weights must be cached on a login node first (see cp3-A). `data/eval/` (68 MB) has to be on the cluster. It isn't on Drive (only `data/debug/` is), so regenerate it there with `python scripts/collect.py --config configs/data/eval.yaml --workers 2` (seeded, so it is identical; D9), or Aunonto can upload this laptop's copy.

## Deviations
- **Branched from `milestone1`, not `cp3`**, like `cp4/B-noiseaug`. `milestone1` is on `main` after `cp3` and has the D21 sampler fix (CPU generator noise on any device), which the rollout relies on for reproducible results.
- **Trajectory start: `start: 32` rows into each episode.** For the first ~32 agent steps after reset the game is in its start-up phase: frames change only every other step with period 4 (48–53% of steps change in rows 0–24, 100% from step 32). Starting at row 0 gave copy-last a perfect 100 dB at h = 20 on all 64 trajectories and identical numbers at h = 1 and 5. Eval episodes have ≥ 262 rows, so 32 + K + T = 236 fits every episode. `start` is a config key.
- **Which trajectories.** One per episode, n spread evenly over the eligible episodes (sorted by shard path, then shard order), as `pick_indices` does in CP3-A. Ground truth is the stored frames, which D9 shows equal a deterministic replay.
- **Generator.** One CPU generator per model and level, seeded with `seed`, is consumed through all steps and batches in order. Results therefore depend on `batch_size` (same as `eval_single_step.py`) but are reproducible across devices (D21).
- **Context levels.** `ctx_levels` (default `[0]`) is swept for checkpoints only. Model tags are `<run>_<ckpt>@ctx<level>`; copy-last runs once.
- **Outputs.** CSV and JSON hold every h = 1..T. JSON also has a `report` entry at `report_horizons` (4.11's list). Videos are of `n_videos` (4) evenly spaced trajectories per model, so the same trajectories for every model. Videos are written with `imageio` v2's ffmpeg writer (libx264, `macro_block_size=1` so frames keep their exact size); no new dependency.
- Generated frames are fed back as floats, not re-quantized to uint8.

## Interface change requests
None.

## Follow-ups
- Run the Discovery command above for the CP3 model (Done criterion), then the full integration with CP4-B's checkpoints.
- If the per-step cost matters for CP6, LPIPS dominates CPU eval time. On a laptop the copy-last run took 163 s, mostly LPIPS on 12,800 frames.
- The h ≈ 128 bump is the same for every model, since it comes from the trajectories' shared game phase. Keep it in mind when reading the main figure. Random `start` offsets per episode would smooth it out but make the start-up phase harder to avoid; not done.
