# cp5-A: Controllability evaluation and inverse dynamics

## Summary
- `wm/eval/detectors.py`: `player_x` (centroid of player-green pixels in raw rows 185–194, NaN when no player is visible), `laser_present` (laser grey in the launch zone just above the player), `calibrate_player_x` (fit against RAM player x), and `to_uint8_frames`. Colours match within a per-channel tolerance of 30, so the detectors work on generated frames.
- `wm/eval/counterfactual.py`:
  - `pick_branches`: one branch row per eval episode, drawn with a seed.
  - `real_branches`: replays the episode exactly, checks that the replay reaches the stored frame, then runs each program from an ALE `cloneState`.
  - `model_branches`: batched `rollout` from the real context, following 4.11 alignment.
  - Metric helpers `ordered`, `mean_dx` and `fire_response`.
- `wm/models/idm.py`: `IDM`, a CNN on `concat(f_t, f_{t+1})` with move (3) and fire (2) heads, plus `action_targets` (from `ACTION_TO_MOVE_FIRE`), `loss_and_accuracy`, `save_idm` and `load_idm`.
- `scripts/eval_counterfactual.py`: real emulator plus any models. Reports ordering accuracy, Δx per step, fire response with a no-fire control, the final-x error against the real game, and IDM accuracy, and writes `counterfactual.png`. `scripts/train_idm.py` trains on the train split and reports accuracy, confusion matrices and majority baselines on the whole val split.
- **Real reference and copy-last on `data/eval`** (64 branches): see Results. **The CP4 aug-on model is not run yet**, because its checkpoint is only on Discovery. The command for Syed is below. That leaves the Done criterion ("results for the CP4 aug-on model") open.

## Files
Created: `scripts/eval_counterfactual.py`, `scripts/train_idm.py`, `configs/eval/counterfactual.yaml`, `configs/train/idm.yaml`, `tests/test_control.py`, `progress/cp5-A.md`.
Modified (Phase 0 stubs replaced): `wm/eval/detectors.py`, `wm/eval/counterfactual.py`, `wm/models/idm.py`.

## Tests
`tests/test_control.py` (26; 4 are `env`, using a small real shard the test writes with `ShardWriter`):
- **Detectors on synthetic frames:**
  - `player_x` is the sprite centroid at x ∈ {8, 19, 42, 75}, including with σ=8 noise, a laser and a bullet present.
  - It returns NaN with no player, and ignores player green outside the band.
  - `laser_present` is true only for a fresh shot above the player. It is false for a streak in another column, a high (old) shot, and frames without a player.
  - Calibration recovers a known linear RAM map.
  - The [-1,1] to uint8 round trip is exact.
- **Mapping table coverage:** `ACTION_TO_MOVE_FIRE` has all 6 actions, is one-to-one, and its move and fire flags agree with the `ACTION_MEANINGS` names. `action_targets` follows it.
- **Branch alignment** (`encode_rows=True` shard):
  - Context frames decode to rows t−K+1..t, and `prefix_actions` equals the stored a_0..a_{t−1}.
  - t is in range, and the seed and `laser_idle` match the stored values.
  - Branches spread over the long-enough episodes, are seeded, and are rejected when there are too few episodes or no ready row.
- **Model branches:**
  - An action-echo model shows g_{t+1} is produced by the program's first action, the first window starts at a_{t−K+1}, and by step K the window is all program.
  - Step 0 is the real branch frame.
  - Copy-last repeats it.
- **Metrics:** the ordering margin, NaN handling and last-step-only rule; Δx skipping missing steps; fire response conditioned on an idle laser and the window, with step 0 excluded, and pooled over programs.
- **Real reference (`env`):**
  - Every program's branch equals a fresh `make_env` replay of prefix plus program, frame for frame.
  - LEFT < NOOP < RIGHT holds on the real game.
  - The branch row is ready (player visible, off the walls, no laser).
  - A branch the replay doesn't reach raises.
- **IDM:** shapes and gradients at 84 and 96; **overfits 256 fake pairs** (random labels, 100% move and fire in 100 full-batch steps); checkpoint round trip.
- **Scripts:**
  - `train_idm.py` end to end on a fake shard: JSON, confusion totals, checkpoint loads.
  - `eval_counterfactual.py` end to end (`env`) with copy-last plus a tiny checkpoint and an untrained IDM: copy-last ordering accuracy is **0**, as the roadmap expects, and its Δx is 0. Real ordering is 1.0 on the scripted episodes, real LEFT Δx < 0 < RIGHT, the IDM pair counts are right, the calibration runs, and the figure is written.

Commands (macOS x86_64 i5, 4 threads, Python 3.12.6, torch 2.2.2, ale-py 0.10.2, CPU):
- `pytest -m "not slow and not gpu and not network"`: 313 passed, 1 skipped (needs CUDA) in 167 s. That's 287 at `cp4` plus 26 new. Earlier runs while my eval jobs were also running took 240–395 s; the slowest tests are pre-existing (`test_loader_num_workers_2` 20 s, `test_wm_never_imports_pygame` 16 s).
- `pytest -m "slow or network or env"`: 28 passed in 303 s (includes this workflow's 4 `env` tests).

## Results
**Detector calibration** (all 39,631 `data/eval/` rows): detected x = 0.5248 · RAM[28] + 0.795. **MAE 0.293 px** over the 37,794 frames with a player (p99 0.70 px; 1,837 frames have no visible player). The max error, 43.6 px, is on frames where the RAM position and the drawn sprite disagree for a step, probably at respawn.

**Laser detector against RAM** (the player-laser byte, see Deviations), on all eval rows from step 32 with a visible player, the laser idle, and the action held for 3 steps. The response is a launch-zone detection at steps 1–3:

| held action | n | detector | RAM launch | agreement |
|---|---|---|---|---|
| FIRE | 277 | 0.523 | 0.487 | 96.4% |
| NOOP | 655 | 0.049 | 0 | 95.1% |
| LEFT/RIGHT | 1,610 | 0.104 | 0 | 89.6% |

Its false positives are alien bullets (same grey) falling through the launch zone, which is why the script reports the no-fire programs as a control.

**Counterfactuals:**

`python scripts/eval_counterfactual.py --config configs/eval/counterfactual.yaml --models copy_last`

64 branches, one per eval episode (t from 34 to 359, median 146), horizon 30, margin 2 px, fire window 3. Real reference: 57 s on CPU, no episode ended.

| source | ordering acc. | Δx/step LEFT | NOOP | RIGHT | fire response (n=192) | no-fire control |
|---|---|---|---|---|---|---|
| real emulator | 0.719 (46/64) | −0.603 | −0.118 | +0.433 | 0.766 | 0.068 |
| copy-last | 0.000 (0.000 on the 46 real-ordered) | 0 | 0 | 0 | 0.047 | 0.047 |
| CP4 aug-on | *pending (Discovery)* | | | | | |

- **The real game's ordering accuracy is 0.72, not 1.** Nearly all remaining failures are life losses in the 30 steps: the player respawns at the left wall (x=19) under every program, which also gives NOOP its negative Δx. For models the script also reports ordering accuracy on the 46 branches where the real game is ordered, which removes that ceiling.
- **The real fire response is 0.77, not 1.** Even with no laser in flight, a held FIRE launches within 3 steps only about half to three-quarters of the time. RAM shows another gate, probably the explosion of the previous shot. Compare the model against the real rate, not against 1.
- Figure: `runs/eval_counterfactual/cp5_copy_last/counterfactual.png` (not in git). `docs/figures/cp5_counterfactual.png` is the integration figure and comes from the trained-model run.

**Inverse dynamics on real val** (`data/debug/` val split, 23,260 transitions; `python scripts/train_idm.py --config configs/train/idm.yaml --set num_workers=3`, 4,000 steps at batch 128, CPU):

*Training on the laptop CPU (about 2.7 s per step with lazy HDF5 loading; started 2026-11-18). Numbers go here when it finishes. The Discovery job below produces the same table in a few minutes.*

**For Syed (Discovery).** One job trains the IDM, and the second runs the counterfactuals on the aug-on checkpoint, scoring them with that IDM. `num_steps=2` is D11; the CP4 checkpoints store 3. `cfg_scale` stays at the checkpoint default (`w`=1) until CP5-C's sweep picks a weight; the integration check reruns the second command with `--set cfg_scale=<w>` and the CP5-B checkpoint.

```bash
D=/dartfs-hpc/scratch/f0079km/wm/data
jid=$(CMD="python scripts/train_idm.py --config configs/train/idm.yaml --set dataset_dirs=[$D/debug] --set num_workers=6" \
  sbatch --parsable --partition=gpuq --time=00:30:00 scripts/slurm/eval.sbatch)
CMD="python scripts/eval_counterfactual.py --config configs/eval/counterfactual.yaml \
  --models copy_last runs/20261117-0431_debug_aug/checkpoints/ckpt_last.pt \
  --set dataset_dirs=[$D/eval] --set num_steps=2 --set idm=runs/idm/idm.pt --set out_dir=runs/eval_counterfactual/cp5_aug" \
  sbatch --dependency=afterok:$jid --partition=gpuq --time=00:30:00 scripts/slurm/eval.sbatch
```

The cost is small on a GPU:
- IDM: 4,000 steps of a 1.3M-parameter CNN at batch 128.
- Counterfactuals: 384 rollouts × 30 steps × 2 sampler steps, batch 32, plus about 1 min of emulator replay and 1 min of calibration.

`counterfactual.png` from the second job is the CP5 figure (rename it to `docs/figures/cp5_counterfactual.png`). The emulator runs headless (ale-py is in `requirements.txt`).

## Deviations
- **Branch rows are "ready" rows.** t ≥ max(K, `min_step`=32, after the start-up steps in which the game ignores input), with t + 30 inside the stored episode. The player must be visible, with RAM player x at least 8 units (about 4 px) from both walls, and no player laser in flight. t is uniform over the ready rows of each episode (`branch_seed` 0).
  - With unrestricted rows only 15 of 64 branches had an idle laser, too few for the fire metric.
  - With only the laser condition, real ordering accuracy was 0.50: a third of the failures were a player against a wall, where LEFT or RIGHT can't move it.
  - Deaths are kept, since they are part of the game.
- **"No laser on screen" comes from RAM, at the branch row only.** The player's laser and alien bullets are the same grey, and pixels can't tell them apart. The branch row is real for both the model and the emulator, so this doesn't favour either. Byte **85** is the player laser's y: it reads **246** when no shot is in flight and falls by 4 per step from about 85 during one. I found it in CP5-A by correlating RAM with a scripted FIRE run, then checked it on the eval data (table above). The constant is in `counterfactual.py`, not `wm/data/ram.py`, which CP2-A owns. Moving it there is a follow-up. RAM is never a model input (D8).
- **The fire response is pooled** over FIRE, LEFTFIRE and RIGHTFIRE (n = 3 × 64). The pooled NOOP, LEFT and RIGHT rate is reported as the detector's false-positive control.
- **The response detector looks at the launch zone** (raw rows 160–184, rows 64–73 at 84, columns x−2..x+3 around the player). A new shot is always there on the step after FIRE. Whole-screen grey would count alien bullets and lasers already in flight.
- **Ordering at step 30 uses the 2-px margin on both sides:** x_L + 2 < x_N and x_N + 2 < x_R. A missing player (NaN) under any of the three programs counts as a failure. Δx per step is the mean over steps where both frames show the player.
- **IDM on model rollouts uses the counterfactual branches**, whose actions are known (each program, all 30 steps). It scores every pair (s_j, s_{j+1}) for the model and for the real emulator on the same branches. The real number is the matched ceiling.
- **IDM first layer is stride 1.** The player moves only about 0.5 px per step at 84, so early downsampling would lose the signal.
- **Two new configs** (`configs/eval/counterfactual.yaml`, `configs/train/idm.yaml`) aren't in the Owns list. 4.12 needs `--config`. Same as CP4-A and CP5-C.
- `eval_counterfactual.py` reuses `build_models` from `scripts/eval_rollouts.py`, so copy-last and the checkpoints load and sweep `ctx_levels` exactly as in the CP4 eval.

## Interface change requests
None. The RAM laser byte (85, idle value 246) could be added to `wm/data/ram.py` by its owner.

## Follow-ups
- **Done criterion:** run the Discovery commands above for the CP4 aug-on model, then fill in the table. Then do the CP5 integration check: the CP5-B checkpoint, CP5-C's `w`, and `docs/figures/cp5_counterfactual.png`.
- The model's fire response depends on it drawing the shot in the first 3 steps. If it draws the shot one step late, widen `fire_window` rather than reading it as no control.
- On the real branches the player is missing on 4.4% of program steps (deaths and respawns). `player_missing` in the JSON reports this per source, so a model that loses the player sprite shows up there.
