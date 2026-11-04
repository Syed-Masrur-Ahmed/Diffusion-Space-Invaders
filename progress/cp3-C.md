# cp3-C: U-Net denoiser

## Summary
- `UNet(UNetConfig)` implements ROADMAP 4.7 with the `[FROZEN]` forward signature. Input `cat([x, context])` goes through conv3×3, AdaGN ResBlocks (`n_res` per level, stride-2 conv downsampling, nearest ×2 upsampling plus conv3×3, skip concatenation), and a 2-ResBlock middle with optional self-attention. The output head is GroupNorm → SiLU → zero-initialized conv3×3, so F = 0 exactly at init.
- `wm/models/embeddings.py`: `FourierFeatures` (16 fixed frequencies, scale 16, sin and cos) and `ConditionEmbedding`, which builds `emb = SiLU(noise + actions + ctx_level)` exactly as 4.7 lists.
- `configs/model/small.yaml` (14.3M parameters at 84) and `configs/model/base.yaml` (34.4M at 84). Each holds `ch` for both 84 and 96. `unet_config(model_dict, C, K, H)` builds the `UNetConfig`.
- The model prints its parameter count at construction. Invalid `H` raises a `ValueError` that names `H`, the required factor, and `ch`.

## Files
Created:
- `tests/test_unet.py`
- `configs/model/small.yaml`, `configs/model/base.yaml`
- `progress/cp3-C.md`

Modified:
- `wm/models/embeddings.py` (was a docstring only): `FourierFeatures`, `ConditionEmbedding`
- `wm/models/unet.py`: `UNetConfig` fields unchanged; adds `unet_config`, `AdaGN`, `ResBlock`, `SelfAttention`, `UNet`

## Tests
Added `tests/test_unet.py` (36 fast tests, plus 1 `gpu` test):
- output shape and dtype for H ∈ {84, 96} × K ∈ {1, 4, 8} × C ∈ {1, 3}, using the 4.7 `ch` lists and `n_res=2`
- output is exactly zero at init, with and without attention
- after perturbing every zero-initialized parameter (see Deviations), changing `x`, `c_noise`, `context` (oldest and newest frame), `actions` (oldest and newest), or `ctx_level` (oldest and newest) changes that sample's output and leaves the other sample in the batch unchanged
- every parameter gets a gradient: at init all `.grad` exist; after the perturbation every gradient is non-zero (with and without attention)
- indivisible H raises `ValueError` (84 with 4 levels, 90 with 3, 16 with 6)
- dropout is active in `train()` and inactive in `eval()`
- a `state_dict` round trip into a differently seeded model reproduces the output, so the Fourier frequencies are saved
- the parameter count is printed
- `small` and `base` at 84 and 96 build and run. `small` matches the 4.7 `ch` lists with `n_res=2`. Each 96 list is the 84 list plus one more level at the lowest width. Parameter targets (small 10–15M, base 30–40M) are checked at 84.
- `unet_config` picks `ch` by H, accepts a `--set ch=[...]` list override, and raises `KeyError` for a resolution that is not in the map
- inside `EDMDenoiser`: with a 5-D context, `denoise` at init equals `c_skip·x`, and `loss` is finite and backpropagates
- `gpu`: forward time of `small` at 84, batch 64, in fp32 and bf16 autocast (prints ms; asserts finite output and < 1 s)

Commands (Apple M3, macOS arm64, Python 3.13.16, torch 2.14.1, CPU):
- `pytest tests/test_unet.py` → 36 passed, 1 skipped (CUDA not available), in about 5 s
- `pytest -m "not slow and not gpu and not network"` → 201 passed, 5 deselected, in 28 s (165 at cp2). Inside Cursor's sandbox, 4 `test_dataset.py` worker tests fail because `torch_shm_manager` is blocked. Outside the sandbox they pass.
- `pytest -m "slow or network or env or gpu"` → 20 passed, 1 skipped (`test_forward_time_batch64`, no CUDA), in 20 s

**GPU test, for Syed to run on Discovery** (no config needed; it reads `configs/model/small.yaml`). Add the partition and account options from section 1.4:
```bash
sbatch --gres=gpu:1 --time=00:15:00 --output=runs/slurm-unet-gpu-%j.out \
  --wrap "source .venv/bin/activate && pytest -m gpu tests/test_unet.py -s -q"
```
The fp32 and bf16 times print to the log (lines starting `UNet small@84 forward`).

## Results
Parameter counts (C=3, K=4; C and K change them by under 0.1%):

| config | H | ch | parameters |
|---|---|---|---:|
| small | 84 | [64, 128, 256] | 14,263,683 |
| small | 96 | [64, 128, 256, 256] | 22,922,627 |
| base | 84 | [128, 256, 384] | 34,352,387 |
| base | 96 | [128, 256, 384, 384] | 52,845,827 |

Forward time at 84, fp32, `inference_mode`, batch 64 unless noted, on an Apple M3 laptop (4 torch threads):

| config | CPU | MPS |
|---|---:|---:|
| small | 5.78 s | 0.95 s |
| base | 16.4 s | 2.51 s |
| small, batch 1 | 60 ms | 14 ms |

`small` costs about 9 GMAC per sample at 84×84, so expect tens of ms per batch-64 forward on a datacenter GPU. The real number comes from the `gpu` test above.

Sanity run (not committed): `EDMDenoiser(UNet(small))` on one fixed batch of 16 synthetic sparse-sprite frames (84×84×3, K=4), with the same σ and ε each step, AdamW 3e-4, on MPS. Loss went from 3.07 to 0.55 (5.6×) in 300 steps, with no NaN.

## Deviations
- **Sensitivity and gradient tests perturb every zero-initialized parameter, not only the output conv.** The team approved this before implementation. Under 4.7 the AdaGN `Linear(emb_dim, 2C)` layers are zero-initialized, and `emb` is the only path from `actions`, `c_noise`, and `ctx_level` to the output. So re-initializing only the output conv, as section 5 says, cannot make those inputs change the output. The zero-inits also make every gradient upstream of the output conv exactly zero at init. The tests therefore give every all-zero parameter small random values (std 0.05): output conv, ResBlock second convs, AdaGN Linears, attention output projection, the output GroupNorm bias, and the action positional embedding. Exact zero output at init is tested separately on the unmodified model.
- **Parameter counts at 96 are above the 4.7 targets.** The 96 list `[64,128,256,256]` that 4.7 fixes adds a fourth level, so `small` is 22.9M (target ≈10–15M). `base` follows the same pattern and is 52.8M. The explicit channel lists win over the approximate counts, so the test checks the targets at 84 (D10) only. If 96 is ever used, a narrower `ch` can be passed with `--set`.
- **Model config layout.** A model YAML holds the `UNetConfig` fields except `C`, `K`, `H`, which come from the data config. `ch` is a map from resolution to channel list. `unet_config(model, C, K, H)` picks the list and builds the dataclass, raising `KeyError` on unknown keys or a missing resolution. A plain list is also accepted, so `--set model.ch=[32,64,128]` works. Overriding one entry of the map (`model.ch.84=...`) does not, because `load_config` compares the key as the string `"84"`.
- **Architecture details 4.7 leaves open:**
  - Skips are the outputs of the encoder ResBlocks (one per block, `n_res` per level), not `conv_in` or the downsample outputs. Each decoder level has `n_res` ResBlocks, each concatenating one skip.
  - Down and up convs keep the channel count. Width changes in the first ResBlock of a level, through its 1×1 skip conv.
  - With `attn_lowest`, a single self-attention block sits between the two middle ResBlocks. It uses GroupNorm with affine weights, one 1×1 qkv conv, heads = C/64 (or 1 if C is not a multiple of 64), PyTorch `scaled_dot_product_attention`, and a zero-initialized 1×1 output conv as a residual. It is not conditioned on `emb`.
  - The output head's GroupNorm (plain GroupNorm, not AdaGN, as 4.7 says) has affine weights.
  - Fourier features follow EDM's `FourierEmbedding`: `freqs ~ N(0, 16²)` stored as a buffer (saved in checkpoints, not trained), and `[cos, sin](2π·freqs·c_noise)` computed in float32, giving 32 features.
  - The action positional embedding `(K, 64)` is initialized to zero.
- An indivisible `H` raises `ValueError` instead of a bare `assert`, so the check survives `python -O`.

## Interface change requests
None.

## Follow-ups
- **CP3-B (trainer):** build the model with `UNet(unet_config(load_config("configs/model/small.yaml"), C, K, H))`. If the model block is nested in the train config, pass `cfg["model"]` instead. `C`, `K`, and `H` come from the data config (`H` = D10 resolution 84; C = 3 for RGB, 1 for greyscale). The constructor prints the parameter count, so `copy.deepcopy` for EMA does not print again. Put `--set model.ch=[...]` overrides on the whole list.
- **CP3-B / integrator:** run the `gpu` timing test on Discovery (command above) and record the time here or in section 1.5.
- **CP4-C:** `channels_last` and `torch.compile` are untested on this model. The attention path uses `scaled_dot_product_attention`, which should compile. Measured batch-1 forward of `small` at 84 on the M3: 60 ms on CPU and 14 ms on MPS. At 3 Euler steps that is about 5.5 and 24 frames/s.
- **CP5-C:** the `action_input` flag goes on `UNetConfig`. The `"adagn+planes"` input planes would widen `conv_in` (now `C·(K+1)` input channels). The action embedding lives in `ConditionEmbedding`.
