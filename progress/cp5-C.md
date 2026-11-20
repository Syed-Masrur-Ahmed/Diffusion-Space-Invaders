# cp5-C: Action conditioning and classifier-free guidance

## Summary
- Action CFG was already implemented in `_guided_denoise`. `w = 1` is one forward on the real actions. Any other `w`, including `0`, is one doubled forward: `D(x; ∅) + w (D(x; a) - D(x; ∅))`, with the whole action sequence replaced by `NULL_ACTION` on the null half. `w = 0` matches sampling with every action set to `NULL_ACTION`. The same `cfg_scale` reaches `sample`, `WorldModel.predict_next`, and `rollout` (`num_steps`, `cfg_scale`, `ctx_level`, `generator`).
- `UNetConfig.action_input` is `"adagn"` (unchanged 4.7 conditioning) or `"adagn+planes"`. The planes variant concatenates a 7-channel one-hot of the latest action, including `NULL_ACTION`, as constant input planes, which widens `conv_in`. The forward signature is unchanged. Output is still 0 at init. `WorldModel.from_checkpoint` reads `model.action_input` from the checkpoint; a missing key stays `"adagn"`, so CP4 checkpoints still load.
- `scripts/action_sensitivity.py` scores val windows by the mean absolute difference, in 8-bit pixel units, between LEFT and RIGHT predictions in the player band. Both calls use one noise seed. It sweeps `w`, writes a copy-last row (always 0), grids, and proposes the smallest `w` that reaches 80% of the rise from the smallest `w`.
- **Sweep run on Discovery (2026-11-20)** for the CP4 aug model and the CP5-B final checkpoint. Neither ignores actions (LEFT/RIGHT band MAE ≈ 2.0 at `w` = 1, copy-last 0), so `adagn+planes` is not needed and stays untrained. **D11 CFG weight: `w` = 1** (Syed, 2026-11-20). The script's proposal (`w` = 3) was rejected: the sensitivity score grows with `w` by construction, and the counterfactual eval at `w` ∈ {1, 1.5, 2, 3} shows no control gain and a worse fire response for `w` > 1. See Results.

## Files
Created:
- `tests/test_cfg.py`
- `scripts/action_sensitivity.py`
- `configs/eval/action_sensitivity.yaml` (section 3.4 allows a new config; 4.12 requires `--config`)
- `progress/cp5-C.md`

Modified:
- `wm/diffusion/sampler.py` (docstring of `_guided_denoise` only; the CFG math is unchanged)
- `wm/inference.py` (`from_checkpoint` builds `adagn+planes` when the checkpoint records it)
- `wm/models/unet.py` (`UNetConfig.action_input`)

## Tests
Added in `tests/test_cfg.py` (17):
- `w = 1` matches a call that omits `cfg_scale`, and the denoiser sees one batch of the real actions
- `w = 0` matches all-`NULL_ACTION` at `w = 1` (closed form `0.1 * NULL_ACTION` on a one-step action oracle) and still sends the doubled batch, null half all `NULL_ACTION`
- the same two facts through `WorldModel.predict_next` and `rollout`; `w = 2` doubles the batch, `w = 1` does not
- `WindowDataset` windows never contain `NULL_ACTION` or a negative action, for `K` in {1, 4, 8} and both splits
- `action_input` defaults to `adagn`; an unknown value raises
- `adagn+planes`: output shape for `(C, K, H)` in {(3, 4, 16), (3, 4, 84), (1, 1, 16)}, `conv_in` grows by 7 channels, parameter delta is the extra 3×3 kernel only, output is 0 at init, the planes are the one-hot of `actions[:, -1]` and are constant in space, an older action does not change them
- every parameter gets a gradient, including the new `conv_in` slice; changing the latest action or an older action changes the output
- a planes checkpoint round-trips through `from_checkpoint`, LEFT differs from RIGHT, and the channels-last default matches the contiguous load (atol 1e-4); an explicit `adagn` checkpoint keeps the old `conv_in` width
- player band is raw rows 185–195 scaled onto the frame (84 → [74, 78)); band MAE is in 8-bit units and ignores rows outside the band
- `propose_cfg_weight` picks the smallest `w` past 80% of the rise, and the smallest `w` with a reason when guidance does not help
- the yaml lists `w` in {1, 1.5, 2, 3}; the script on a fake shard writes json, csv, and `grid_w1.png`, copy-last sensitivity is 0, and a tiny net that reads actions is above 0

Commands (macOS arm64, Python 3.13.16, torch 2.14.1; MPS available, CUDA not):
- `pytest -m "not slow and not gpu and not network"` → 305 passed, 8 deselected, in 42 s (288 at the cp4 tag)
- `pytest -m "slow or network or gpu"` → 7 passed, 1 skipped, in 101 s. The skip is `tests/test_unet.py::test_forward_time_batch64` (CUDA not available). This workflow adds no `slow`, `gpu`, or `network` tests.

## Results
**Sensitivity sweep (Discovery, one GPU, `num_steps` 2, n = 64 val windows, mean ± std, 8-bit MAE in the player band):**

| `w` | CP4 aug (`20261117-0431_debug_aug`, `$D/debug` val) | CP5-B final (`20261118-0501_final`, `$D/full` val) |
|---|---|---|
| copy-last | 0.000 | 0.000 |
| 1 | 2.021 ± 0.968 | 1.973 ± 1.001 |
| 1.5 | 2.567 ± 1.232 | 2.477 ± 1.248 |
| 2 | 3.075 ± 1.475 | 2.989 ± 1.499 |
| 3 | 3.878 ± 1.865 | 3.749 ± 1.891 |

Script proposal: `w` = 3 for both (`ignores_actions` false). Outputs in `runs/action_sensitivity/{cp4_aug,final}/` (not in git).

**Why the proposal is not adopted.** LEFT and RIGHT share the noise seed, so `D(x; ∅)` cancels in their difference and the guided gap is roughly `w × (D(a_L) − D(a_R))`: the score rises with `w` for any model that reads actions, and the 80% rule then always picks the largest `w` in the sweep. A larger LEFT/RIGHT gap is not better control. The deciding test is the CP5-A counterfactual eval (`eval_counterfactual.py`, final checkpoint, same 64 branches, `--set cfg_scale=<w>`):

| `w` | ordered (of 36 real-ordered) | Δx/step LEFT | NOOP | RIGHT | fire response | IDM move | IDM fire | model time |
|---|---|---|---|---|---|---|---|---|
| **1** | 34 | −0.383 | −0.023 | +0.389 | **0.599** | **0.681** | 0.571 | 53 s |
| 1.5 | 35 | −0.388 | −0.004 | +0.388 | 0.583 | 0.679 | 0.574 | 112 s |
| 2 | 35 | −0.399 | −0.003 | +0.388 | 0.552 | 0.675 | 0.573 | 112 s |
| 3 | 36 | −0.406 | −0.002 | +0.408 | 0.521 | 0.674 | 0.570 | 112 s |
| real emulator | 36 | −0.614 | −0.250 | +0.267 | 0.625 | 0.689 | 0.575 | |

- Movement is saturated at `w` = 1: Δx changes by ≤ 0.02 px/step and ordering by 2 of 36 branches (noise at this n).
- Fire response falls with `w` (0.60 → 0.52, real 0.625), IDM move falls slightly, and NOOP moves further from the real drift.
- Any `w` ≠ 1 doubles the denoiser batch, so it costs 2× per frame.
- Outputs: `runs/eval_counterfactual/final_w{1.5,2,3}/`; the `w` = 1 row is CP5-A's `runs/eval_counterfactual/cp5_aug/`. The player band used by the script is raw rows 185–195 (the green cannon, not the score digits; same rows as `scripts/coverage_report.py`), which is rows [74, 78) at 84×84.

`adagn+planes` adds `(NUM_ACTIONS + 1) * ch[0] * 9` parameters, the extra `conv_in` kernel. At `ch[0] = 64` that is 4,032 weights. The bias is unchanged. The `small` and `base` counts at `"adagn"` are unchanged.

**D11 CFG weight: `w = 1`** (one forward, no guidance), the code default, so nothing changes in code or configs. The proposal rule, once the sweep exists, is the smallest `w` in {1, 1.5, 2, 3} whose sensitivity reaches 80% of the rise from `w = 1` to the best `w` in the sweep. Adopt it only after looking at `grid_w<w>.png` (last context | LEFT | RIGHT | |diff|×4). If the best sensitivity is within `ignore_below` (0.5 8-bit MAE) of copy-last, the script reports that the model ignores actions and does not propose a `w`.

For Syed, from the repo root, on the CP4 aug checkpoint. `num_steps` in the yaml is already 2 (D11). Point `data.dataset_dirs` at scratch if `data/debug` is not next to the repo. `eval.sbatch` requests one GPU.

```bash
CMD="python scripts/action_sensitivity.py --config configs/eval/action_sensitivity.yaml --set model=runs/20261117-0431_debug_aug/checkpoints/ckpt_last.pt --set data.dataset_dirs=[/dartfs-hpc/scratch/f0079km/wm/data/debug]" \
  sbatch --partition=gpuq --time=00:20:00 scripts/slurm/eval.sbatch
```

Training `adagn+planes` would need the (withdrawn) trainer change in Interface change requests; not planned. After that lands, a debug run is `configs/train/debug_aug.yaml` plus `model.action_input: adagn+planes` (same steps, seed, and `aug_max` 0.7).

## Deviations
- **CFG math was not rewritten.** CP2-C already had the doubled batch, the `w = 1` short circuit, and `w = 0` equal to all-`NULL_ACTION`. CP5-C locks that path through `WorldModel` and `rollout` and documents it on `_guided_denoise`.
- **`action_input` was added before knowing whether the aug model ignores actions.** The tests require the variant, and the CP4 carry-over says the flag goes on `UNetConfig`. There is no checkpoint here to measure sensitivity, so nothing was retrained.
- **New config file.** `configs/eval/action_sensitivity.yaml` is not in the Owns list. Section 3.4 says a workflow may add configs. The sweep weights, the band, the 80% gain, and the 0.5 ignore threshold live there rather than in the script.
- **What "the bottom band" and "LEFT vs RIGHT" mean.** The band is the player sprite's raw rows, scaled like the coverage bands, not a fixed fraction of the frame. Only `actions[:, -1]` is replaced. That is the action that produces the next frame (4.5). The older actions stay, because they belong to the real context frames. The score is the mean of `|LEFT - RIGHT| * 127.5` over the band's pixels and channels.
- **Proposal threshold.** "Most of the sensitivity gain" is 80% of the rise from the smallest `w` (1) to the best `w`. The script cannot see artifacts; `check_grids` is set when it proposes a `w`.
- **Loading `action_input` without a trainer field.** `ModelSection` does not have the key, and `from_dict` rejects unknown keys. `from_checkpoint` copies it off the dict, then builds the U-Net with it. Absent means `"adagn"`, which is `build_net`'s existing path. The frozen positional signature of `from_checkpoint` and `predict_next` is unchanged.

## Interface change requests
**Withdrawn (2026-11-20):** the sweep shows both models read actions, so no `adagn+planes` retrain is planned. Kept for reference: `wm/train/trainer.py` would need an additive `action_input` so a retrain can record the flag. It is not a `[FROZEN]` signature. Suggested change, default `"adagn"` so every current yaml and checkpoint is unchanged:

- `ModelSection.action_input: str = "adagn"`
- `build_net` passes `action_input=cfg.action_input` into `UNetConfig`
- `load_run_config` rejects any value other than `"adagn"` or `"adagn+planes"`

Until that lands, `train.py --set model.action_input=adagn+planes` fails: the key is not in the train yamls, and `from_dict` would reject it if it were. `WorldModel.from_checkpoint` already accepts a checkpoint that contains the key.

## Follow-ups
- **D11 (integrator):** record `w` = 1 in ROADMAP D11. CP5's integration figure is the `w` = 1 counterfactual run (`runs/eval_counterfactual/cp5_aug/counterfactual.png` → `docs/figures/cp5_counterfactual.png`); no rerun needed.
- **`propose_cfg_weight` is not a usable selection rule** (it picks the sweep's largest `w` for any action-reading model). CP6-C's CFG ablation should report controllability (CP5-A's counterfactual metrics) per `w`, not this score alone.
- `adagn+planes` and the trainer field below are not needed for this project; the models read actions through `adagn`.
- **CP5-A / CP6-C:** controllability and the CFG ablation can call this script or read its json. The sweep weights match the CP6-C list.
- **Heun at N=2** is still the broken schedule from CP4 (the correction at `sigma_min`). Not touched here. Don't use Heun below 5 steps.
- CP4 checkpoints store `num_steps: 3`. This script's yaml passes 2, so it does not need an extra `--set` for those checkpoints. A caller who overrides `num_steps` back to null will get 3.
