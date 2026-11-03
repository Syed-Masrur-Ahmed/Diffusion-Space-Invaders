# cp2-C: EDM core and sampler

## Summary
- `EDMDenoiser` implements ROADMAP 4.6: preconditioning (`edm_coefficients`), log-normal σ sampling clamped to `[2e-3, 20]`, the `F_target` loss, and `denoise`. Context frames go to the net unscaled, flattened to `(B, K*C, H, W)`.
- `loss_terms` returns per-sample losses and their σ, and `loss_by_sigma` turns them into mean loss per log-spaced σ bucket for trainer logging. `loss` is their mean, so its signature is unchanged.
- `karras_sigmas` and `sample` implement 4.9: Euler (default) or Heun, with the Heun correction skipped on the step to σ=0, and output clamped to [-1, 1]. Action CFG (4.8) uses one doubled-batch call per denoiser evaluation when `cfg_scale != 1`, and the plain single pass when `cfg_scale == 1`.
- `configs/train/edm_default.yaml` holds the `edm:` and `sampler:` defaults, and the tests check they match the dataclasses.

## Files
Created:
- `tests/test_edm.py`, `tests/test_sampler.py`
- `configs/train/edm_default.yaml`
- `progress/cp2-C.md`

Modified:
- `wm/diffusion/edm.py` (`EDMConfig` unchanged; adds `edm_coefficients`, `EDMDenoiser`, `loss_by_sigma`)
- `wm/diffusion/sampler.py` (`SamplerConfig` unchanged; adds `karras_sigmas`, `sample`)

## Tests
Added (all fast, CPU, using `TinyDenoiserNet` or analytic oracles, no U-Net):
- `test_edm.py` (13):
  - coefficient limits as σ→0 (`c_skip → 1`, `c_out → 0`), and exact values at σ=0.5 and σ=2
  - with an oracle net returning `F_target` (float64), the loss is 0
  - at init (zero output conv), `D = c_skip·x` exactly
  - wiring: the net receives `c_in·x`, `ln σ / 4`, the flattened context unscaled, and the actions and levels unchanged; a 4-D and a 5-D context give identical results
  - σ distribution: ln σ has mean -0.4 and std 1.2 (±0.03 over 100k draws), stays within the clamp, and the clamp is hit when `p_std` is large
  - `loss == loss_terms(...)[0].mean()`, and backward reaches every parameter
  - the loss equals EDM's `λ(σ)·mean((D − y)²)`, as 4.6 states
  - `loss_by_sigma`: bucket edges, labels, and means; empty buckets are skipped
  - the YAML `edm:` section equals `EDMConfig()`
- `test_sampler.py` (21):
  - Karras schedule for N ∈ {1, 2, 3, 10}: length N+1, strictly decreasing, first value σ_max, last nonzero value σ_min, ends at 0; N=1 gives `[σ_max, 0]`; N=3 matches the formula
  - Gaussian oracle `D = x·s²/(σ²+s²)` with s=0.3: 50 Heun steps give a std within 3% of s
  - Gaussian oracle, 3 Euler steps: output finite, std < s, and within 3% of the closed-form Euler value (see Deviations)
  - `cfg_scale=1`: exactly one net call per denoiser evaluation, batch not doubled (3 calls for Euler, 5 = 2N−1 for Heun at N=3), counted by a wrapping net
  - `cfg_scale=2`: one call per step with batch 2B
  - the CFG combination `D_u + w(D_c − D_u)` is exact, checked through a 1-step sample, which returns D itself
  - `cfg_scale=0` equals sampling with all-`NULL_ACTION` actions at `cfg_scale=1`
  - `cfg_scale=1` equals a hand-written Euler loop over `denoise`
  - a seeded generator repeats exactly (Euler and Heun); a different seed differs
  - output shape `(B,C,H,W)` and range for (C,K) ∈ {(1,1), (3,4), (3,8)}, with a 4-D or 5-D context
  - an unknown method raises; the YAML `sampler:` section equals `SamplerConfig()`

Commands (Python 3.13.16, torch 2.14.1, macOS arm64, CPU):
- `pytest tests/test_edm.py tests/test_sampler.py` → 34 passed in about 1 s
- `pytest -m "not slow and not gpu and not network"` → 119 passed, 3 deselected, in 15 s (85 at cp1)
- This workflow has no `slow`, `gpu`, or `network` tests.

## Results
Gaussian oracle, s = 0.3, 64×1×16×16 samples, seed 0 (std of samples / s):

| sampler | measured | closed form |
|---|---:|---:|
| Heun, N=50 | 1.0033 | 1.0005 |
| Euler, N=3 | 0.535 | 0.532 |
| Euler, N=10 | 0.823 | 0.820 |
| Heun, N=3 | 2.50 (after the [-1,1] clamp) | 3.75 |

The default schedule at N=3 is σ = [5.0, 0.283, 0.002, 0].

Sanity run, not committed: `EDMDenoiser(TinyDenoiserNet(hidden=32))` on random binary 16×16 frames (target = last context frame rolled by 1 px), batch 32, AdamW 2e-3, 200 CPU steps. Mean loss over the first 20 steps 2.45, over the last 20 steps 0.46, no NaN. `loss_by_sigma` returned five non-empty buckets. `sample` with `cfg_scale=1.5` returned (4,3,16,16) in [-1,1].

## Deviations
- **N=3 Euler oracle test** (approved by the team before committing). The roadmap asks that the std be within 15% of s. That cannot pass under the 4.9 schedule: on Gaussian data Euler shrinks the std at every step, by the cosine of the angle between `(s, σ_i)` and `(s, σ_{i+1})`. The final 0.002 → 0 step does almost nothing, so two steps carry the whole range, and the best value over any s is about 0.53·s. The test now asserts finite output, std < s, and std within 3% of the closed-form Euler value. That still catches a wrong step formula, schedule, or noise scale.
- **Context shape.** `denoise`, `loss`, `loss_terms`, and `sample` accept the context as `(B,K,C,H,W)`, as the dataset and `predict_next` produce it, or already flattened to `(B,K*C,H,W)`. A 5-D context is flattened inside `EDMDenoiser`. 4.6 does not fix the shape.
- **Extra public helpers** (none are `[FROZEN]`):
  - `edm_coefficients(sigma, sigma_data)`
  - `EDMDenoiser.sample_sigma(n, device, dtype)`
  - `EDMDenoiser.loss_terms(...) -> (per_sample_loss (B,), sigma (B,))`, so the trainer can log loss by σ without a second forward pass
  - `loss_by_sigma(losses, sigma, n_buckets=8, sigma_min=2e-3, sigma_max=20) -> {"lo-hi": mean}`: log-spaced buckets with labels in `.3g` format; σ at or outside the range goes to the end buckets
- **`sample` runs under `torch.no_grad()`.** It does not call `eval()` or choose EMA weights; the caller (trainer sample grids, `WorldModel`) does both.
- **Unconditional branch of CFG.** All K actions are replaced by `NULL_ACTION`, matching the whole-sequence action dropout in 4.8. The context and `ctx_level` are the same in both halves. `cfg_scale=0` still runs the doubled batch.
- **`karras_sigmas` returns float64 on CPU.** The sampler converts the values to Python floats, so they work with any device and dtype of `x`.
- **Random number sources.** `loss_terms` draws σ, then ε, from the global torch RNG. Exact resume therefore depends on the trainer restoring global RNG state, which 4.12 already requires. `sample` draws only the initial noise, from `generator` on `context.device`, so on CUDA pass a CUDA generator.

## Interface change requests
None.

## Follow-ups
- **Integrator:** update the CP2-C test text in ROADMAP section 5. The "N=3 Euler, std within 15%" requirement should become "finite, and within 3% of the closed-form Euler std (≈0.53·s on the default schedule)".
- **CP3-B (trainer):**
  - Call `per, sigma = edm.loss_terms(target, context, actions, ctx_level)`, then `loss = per.mean()` and log `loss_by_sigma(per, sigma)`.
  - Batches from `WindowDataset` can be passed as they are; the 5-D context is flattened inside.
  - Action dropout and context noise belong to the trainer; `EDMDenoiser` applies neither.
  - For sample grids, call `sample(ema_denoiser.eval(), ...)`.
  - `configs/train/edm_default.yaml` has the `edm:` and `sampler:` blocks to copy into `debug.yaml`.
- **CP4-C (D11):**
  - On Gaussian data, Euler at N=3 gives about 0.53 of the true std. On real frames this should show up as blur or faded sprites, so check it in the steps benchmark.
  - Heun at N=3 overshoots badly (3.75× before the clamp) because of the huge first step from σ=5 to σ=0.28. Don't pick Heun at very low N without checking.
- **CP5-C:** the CFG hook is `_guided_denoise` in `wm/diffusion/sampler.py`.
