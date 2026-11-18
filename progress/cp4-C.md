# cp4-C: Inference engine and sampler benchmarks

## Summary
- `WorldModel.from_checkpoint` still loads a 4.12 checkpoint (`ema` or `model`, `strict=True`, optional `torch.compile`) and `predict_next` still samples under `eval()` and `inference_mode`. The network is stored channels-last, and an optional fp16 or bf16 autocast wraps the sampler. The returned frame stays float32, contiguous, and in [-1, 1].
- The frozen positional signature is unchanged: `from_checkpoint(path, device, use_ema, compile)` and `predict_next(context, actions, num_steps, cfg_scale, ctx_level, generator)`. `channels_last` and `amp_dtype` are keyword-only. Batched action CFG is unchanged: `cfg_scale=1` is one forward pass.
- `scripts/bench_sampling.py` times batch-1 frames/s for `num_steps ∈ {1, 2, 3, 5, 10}` × {Euler, Heun} on each requested device, and scores teacher-forced PSNR and LPIPS on val windows with `wm/eval/metrics.py`. It writes `results.json`, `results.csv`, and `quality_vs_speed.png`, and proposes the D11 step count.
- CUDA benchmark done on Discovery (2026-11-17, L40S, `debug_noaug` checkpoint, see Results). The rule proposes **1 Euler step**, but quality *falls* as steps are added, so single-frame PSNR is rewarding the blurry 1-step mean. D11 stays open until the rollout check in Follow-ups is done. Heun at N=2 is broken (11.2 dB).

## Files
Created:
- `scripts/bench_sampling.py`
- `configs/eval/bench_sampling.yaml` (section 3.4 allows a new config; 4.12 requires `--config`)
- `progress/cp4-C.md`

Modified:
- `wm/inference.py`
- `tests/test_inference.py`

## Tests
Added in `tests/test_inference.py` (the CP3 loader tests are unchanged and still pass):
- the frozen positional `from_checkpoint(path, device, use_ema, compile)` still samples
- channels-last is the default: the conv weight and the noisy frame and context arrive channels-last; `channels_last=False` keeps NCHW
- channels-last matches contiguous sampling on `TinyDenoiserNet` (atol 1e-5) and on a perturbed tiny U-Net (atol 1e-4)
- fp16 and bf16 autocast: the network output is that dtype, and `predict_next` returns a finite float32 frame in [-1, 1], repeating exactly with the same generator
- `amp_dtype=float32` raises `ValueError`
- `propose_d11` picks the fewest Euler steps within 0.3 dB of 10 steps at ≥ 15 frames/s, keeps 10 steps when fewer steps fall short of that, and returns a reason when the device, the reference PSNR, or a fast-enough row is missing
- `configs/eval/bench_sampling.yaml` lists the roadmap sweep
- the script on a fake shard writes the json, csv, and plot; `--set quality_n=2` is applied; Euler and Heun at N=2 get different PSNR (N=2 is the first schedule where Heun corrects)

Commands (Apple M3, macOS arm64, Python 3.13.16, torch 2.14.1):
- `pytest -m "not slow and not gpu and not network"` → 255 passed, 8 deselected, in 37 s (242 at the cp3 tag)
- This workflow adds no `slow`, `gpu`, or `network` tests. CUDA is unavailable, so those 8 existing tests were not run (`test_forward_time_batch64`, two LPIPS tests, two agent tests, the storage write-speed test, the trainer resume test with 2 workers, and the tiny collect test).

## Results
Speed of an untrained `small` U-Net, 14,263,683 parameters, batch 1, K=4, 84×84 RGB, channels-last, fp32. One frame is one `predict_next`. Warmup 2, then 5 timed frames. The weights do not change this timing. `quality_n=0`, so there is no PSNR, LPIPS, or plot. Raw json from the run is in `/tmp` and is not committed.

Frames/s on CPU:

| method | 1 | 2 | 3 | 5 | 10 |
|---|---:|---:|---:|---:|---:|
| euler | 16.30 | 8.26 | 5.55 | 3.29 | 1.61 |
| heun | 16.33 | 5.43 | 3.26 | 1.79 | 0.83 |

Frames/s on MPS, measured again after the kernels were already cached:

| method | 1 | 2 | 3 | 5 | 10 |
|---|---:|---:|---:|---:|---:|
| euler | 47.4 | 26.7 | 17.9 | 10.7 | 5.37 |
| heun | 52.4 | 17.8 | 10.7 | 5.92 | 2.83 |

Heun at N=1 does one network evaluation, the same as Euler: the second-order correction is skipped on the step that lands on σ=0. On CPU the two N=1 rates match. On MPS, Heun N=1 measured a few frames/s higher across repeats; that gap is the device, not a shorter sampler.

Euler N=3 on CPU (5.55 frames/s) matches the cp3-C estimate of about 5.5 frames/s from a 60 ms forward. Real time is 15 frames/s. On this M3, CPU reaches it only at N=1. MPS reaches it for Euler N=1, N=2, and N=3 (17.9) and for Heun N=1 and N=2. Euler N=5 is 10.7 on MPS.

Euler N=3, same protocol, channels-last and autocast turned one at a time:

| device | channels_last | amp | frames/s |
|---|---|---|---:|
| cpu | yes | fp32 | 5.25 |
| cpu | no | fp32 | 5.08 |
| cpu | yes | fp16 | 1.93 |
| cpu | no | fp16 | 0.08 |
| mps | yes | fp32 | 18.1 |
| mps | no | fp32 | 22.2 |
| mps | yes | fp16 | 15.4 |
| mps | no | fp16 | 19.0 |

On this Mac, fp32 contiguous is the faster MPS path (22 frames/s, close to the cp3-C estimate of 24). channels-last is about neutral on CPU and slower on MPS. fp16 autocast is slower on both. The engine still defaults to channels-last and fp32, which is what a CUDA conv net usually wants; the benchmark turns either off with `--set`.

**D11 stays open.** The rule is the fewest Euler steps within 0.3 dB PSNR of 10 Euler steps, at ≥ 15 frames/s on the training GPU. Local speed cannot answer it. The CP2 Gaussian oracle (Euler N=3 keeps about 0.53 of the true std; Heun N=3 overshoots before the clamp) and the milestone memorization check (SSIM barely moved after the windows were memorized) both say 3 Euler steps is the blurry one. Whether 5 or 10 steps buy back the 0.3 dB, and whether CUDA stays above 15 frames/s while they do, is the Discovery run below.

For Syed, from the repo root, on the CP3 `small` checkpoint. `eval.sbatch` already requests one GPU. Add the account if the cluster requires one. `data/debug` has to be on the machine (or pass `--set data.dataset_dirs=[...]`).

```bash
CMD="python scripts/bench_sampling.py --config configs/eval/bench_sampling.yaml --set model=runs/20261105-0558_debug_small/checkpoints/ckpt_last.pt" \
  sbatch --partition=gpuq --time=02:00:00 scripts/slurm/eval.sbatch
```

CPU quality on 512 windows is the long part (on this M3, sampling alone is on the order of half an hour, and LPIPS is on top). A CUDA-only quality pass is:

```bash
CMD="python scripts/bench_sampling.py --config configs/eval/bench_sampling.yaml --set model=runs/20261105-0558_debug_small/checkpoints/ckpt_last.pt --set devices=[cuda]" \
  sbatch --partition=gpuq --time=01:00:00 scripts/slurm/eval.sbatch
```

A second CUDA pass with the trainer's precision, and one with NCHW, if the first pass is short of 15 frames/s at the step count that holds the PSNR:

```bash
CMD="python scripts/bench_sampling.py --config configs/eval/bench_sampling.yaml --set model=runs/20261105-0558_debug_small/checkpoints/ckpt_last.pt --set devices=[cuda] --set amp_dtype=bf16" \
  sbatch --partition=gpuq --time=01:00:00 scripts/slurm/eval.sbatch
```

### Discovery CUDA benchmark (2026-11-17)
Submitted with `scripts/slurm/eval.sbatch`, one NVIDIA L40S, `gpuq`. Model `runs/20261117-0431_debug_noaug/checkpoints/ckpt_last.pt` (50k steps, EMA, CP4-B), `devices=[cuda]`, channels-last, fp32, no compile, seed 0. Quality: teacher-forced PSNR and LPIPS on 512 windows from `/dartfs-hpc/scratch/f0079km/wm/data/debug`, ±95% CI. Outputs on Discovery and copied locally (not committed): `runs/bench_sampling/cp4_noaug/{results.json,results.csv,quality_vs_speed.png}`.

| method | steps | frames/s | PSNR (dB) | LPIPS |
|---|---:|---:|---:|---:|
| euler | 1 | 279.9 | 35.97 ± 0.45 | 0.0079 ± 0.0012 |
| euler | 2 | 148.0 | 36.02 ± 0.45 | 0.0078 ± 0.0012 |
| euler | 3 | 100.8 | 34.68 ± 0.42 | 0.0073 ± 0.0012 |
| euler | 5 | 60.9 | 32.69 ± 0.38 | 0.0098 ± 0.0013 |
| euler | 10 | 31.3 | 32.24 ± 0.37 | 0.0105 ± 0.0014 |
| heun | 1 | 277.5 | 35.97 ± 0.45 | 0.0079 ± 0.0012 |
| heun | 2 | 101.1 | **11.18 ± 0.10** | **0.514 ± 0.009** |
| heun | 3 | 61.6 | 26.76 ± 0.27 | 0.0356 ± 0.0028 |
| heun | 5 | 35.0 | 31.55 ± 0.35 | 0.0119 ± 0.0015 |
| heun | 10 | 16.7 | 32.06 ± 0.37 | 0.0108 ± 0.0014 |

- **Speed is not the constraint on CUDA.** Every Euler count and every Heun count clears 15 frames/s. Euler 10 steps is 31 frames/s.
- **`propose_d11` output:** Euler, 1 step (35.97 dB vs the 10-step reference 32.24 dB, 280 frames/s). The rule is "fewest steps within 0.3 dB of 10 steps", and it assumes PSNR rises with steps. Here it falls by 3.7 dB from 1 to 10 steps, so the rule picks the smallest N trivially.
- **Why PSNR falls:** one Euler step from σ_max=5 returns the denoiser's estimate of the mean next frame. Against a single ground-truth frame the mean scores best on PSNR. More steps commit to one sharp sample, which pays for every pixel it gets wrong. This is the same blur that the CP2 Gaussian oracle and the CP3 SSIM check pointed to. LPIPS is flat within the CI from 1 to 3 Euler steps and worse from 5 on. Single-frame PSNR and LPIPS can't settle D11; rollout drift and how the frames look have to.
- **Heun N=2 looks like a bug or an ill-conditioned schedule, not a quality curve.** N=1, 3, 5, and 10 are sane. With `sigma_min=2e-3`, `sigma_max=5`, the N=2 schedule is [5, 0.002, 0]. The only Heun correction is evaluated at σ=0.002 on the Euler estimate from σ=5. The correction slope is (x − D(x))/0.002, and the averaged slope is multiplied by Δσ ≈ 5, so any mismatch in D at σ_min is amplified by ~1,250. That is a hypothesis; it was not checked in code. Heun N=3 (26.8 dB) shows a weaker version of it.

## Deviations
- **Keyword-only arguments on `from_checkpoint`.** `channels_last: bool = True` and `amp_dtype: torch.dtype | None = None` sit after the frozen positional parameters. Existing callers, including `eval_single_step.py`, keep working. `predict_next` gained no parameters. The sampler method is switched by assigning `model.sampler` (`dataclasses.replace(..., method=...)`). `num_steps` passed to `predict_next` still keeps that method. `CopyLastFrame` is untouched.
- **Autocast stays inside `predict_next`.** The sample is cast back to `context.dtype` and made contiguous NCHW before it is returned, so metrics and the float32 tests see the same frames as before.
- **New config file.** `configs/eval/bench_sampling.yaml` is not in the Owns list. Section 3.4 says a workflow may add configs, and the sweep (step counts, devices, the 0.3 dB / 15 frames/s rule, the checkpoint path) has to live in a config rather than in the script.
- **channels-last default.** It is on by default, including on CPU, where this M3 shows no speed gain. Outputs are still contiguous NCHW. `torch.equal` on the conv weights still holds against a contiguous checkpoint; the values match and the strides differ.
- **D11 is not a number yet.** See Results. The proposal function is implemented and tested on synthetic rows.

## Interface change requests
None. The frozen positional signatures in 4.10 are unchanged.

## Follow-ups
- **D11 (integrator):** don't take the 1-step proposal as is. Run CP4-A's `eval_rollouts.py` on the `debug_aug` checkpoint at `--set num_steps=1` and at 3 (the current default, already measured in `progress/cp4-B.md`: 25.3 dB at h=20, LPIPS 0.137 at h=200), compare the videos, and record the pick in sections 1.3 and 1.5. The `propose_d11` rule should probably compare on rollout LPIPS rather than one-step PSNR.
- **Heun N=2:** check the hypothesis above with a unit test on a known Gaussian (`TinyDenoiserNet` or the CP2 oracle). Either fix the schedule for Heun (e.g. skip the correction when the next σ is `sigma_min`) or document that Heun needs N ≥ 5. That touches `wm/diffusion/sampler.py`, which is not a CP4-C file.
- **`torch.compile`:** `from_checkpoint(..., compile=True)` still wraps the net after the channels-last conversion, and the input hook is registered on the outer module. It was not timed. `--set compile=true` on a CUDA job is the way to try it; the first call pays the compile.
- **CP4-A:** `predict_next` takes `num_steps`, `cfg_scale`, and `ctx_level`. Euler vs Heun is `model.sampler = dataclasses.replace(model.sampler, method="heun")` before the call.
- On this M3, leave autocast off. fp16 was slower than fp32 on CPU and on MPS, and fp16 without channels-last on CPU dropped to 0.08 frames/s in one timing. CUDA bf16 is a separate question; the trainer already uses it.
