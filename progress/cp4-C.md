# cp4-C: Inference engine and sampler benchmarks

## Summary
- `WorldModel.from_checkpoint` still loads a 4.12 checkpoint (`ema` or `model`, `strict=True`, optional `torch.compile`) and `predict_next` still samples under `eval()` and `inference_mode`. The network is stored channels-last, and an optional fp16 or bf16 autocast wraps the sampler. The returned frame stays float32, contiguous, and in [-1, 1].
- The frozen positional signature is unchanged: `from_checkpoint(path, device, use_ema, compile)` and `predict_next(context, actions, num_steps, cfg_scale, ctx_level, generator)`. `channels_last` and `amp_dtype` are keyword-only. Batched action CFG is unchanged: `cfg_scale=1` is one forward pass.
- `scripts/bench_sampling.py` times batch-1 frames/s for `num_steps ∈ {1, 2, 3, 5, 10}` × {Euler, Heun} on each requested device, and scores teacher-forced PSNR and LPIPS on val windows with `wm/eval/metrics.py`. It writes `results.json`, `results.csv`, and `quality_vs_speed.png`, and proposes the D11 step count.
- D11 is still open. This machine has no CUDA, and it has no `data/debug` and no trained checkpoint, so there is no PSNR to compare with 10 steps.

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

## Deviations
- **Keyword-only arguments on `from_checkpoint`.** `channels_last: bool = True` and `amp_dtype: torch.dtype | None = None` sit after the frozen positional parameters. Existing callers, including `eval_single_step.py`, keep working. `predict_next` gained no parameters. The sampler method is switched by assigning `model.sampler` (`dataclasses.replace(..., method=...)`). `num_steps` passed to `predict_next` still keeps that method. `CopyLastFrame` is untouched.
- **Autocast stays inside `predict_next`.** The sample is cast back to `context.dtype` and made contiguous NCHW before it is returned, so metrics and the float32 tests see the same frames as before.
- **New config file.** `configs/eval/bench_sampling.yaml` is not in the Owns list. Section 3.4 says a workflow may add configs, and the sweep (step counts, devices, the 0.3 dB / 15 frames/s rule, the checkpoint path) has to live in a config rather than in the script.
- **channels-last default.** It is on by default, including on CPU, where this M3 shows no speed gain. Outputs are still contiguous NCHW. `torch.equal` on the conv weights still holds against a contiguous checkpoint; the values match and the strides differ.
- **D11 is not a number yet.** See Results. The proposal function is implemented and tested on synthetic rows.

## Interface change requests
None. The frozen positional signatures in 4.10 are unchanged.

## Follow-ups
- **Syed / integrator:** run the `sbatch` above and record the D11 step count in section 1.3 and 1.5. The PSNR column is also the check on the CP3 carry-over: whether more than 3 Euler steps clears the background speckle that pulled SSIM down to 0.74.
- **`torch.compile`:** `from_checkpoint(..., compile=True)` still wraps the net after the channels-last conversion, and the input hook is registered on the outer module. It was not timed. `--set compile=true` on a CUDA job is the way to try it; the first call pays the compile.
- **CP4-A:** `predict_next` takes `num_steps`, `cfg_scale`, and `ctx_level`. Euler vs Heun is `model.sampler = dataclasses.replace(model.sampler, method="heun")` before the call.
- On this M3, leave autocast off. fp16 was slower than fp32 on CPU and on MPS, and fp16 without channels-last on CPU dropped to 0.08 frames/s in one timing. CUDA bf16 is a separate question; the trainer already uses it.
