# cp1-A: Environment module and resolution decision

## Summary
- `wm/env.py`: `to_model_frame(method="max")` added (`adaptive_max_pool2d` per channel; torch is imported lazily so the play scripts don't load it). `MaxPoolSkip` and `make_env` already matched 4.3; they are unchanged and now pinned down by tests.
- `scripts/inspect_frames.py`: writes `docs/figures/cp1_frames.png` (raw frame plus {84, 96} × {area, max} × {RGB, grey}, 4× nearest, with laser crops); `--laser-stats` prints laser visibility per version.
- **D10 decision: 84×84, `method="max"`.** `area` fails the 95% rule at both 84 and 96 (65.5%); `max` keeps the laser in 100% of frames at both.

## Files
Modified: `wm/env.py`, `tests/test_env.py`.
Created: `scripts/inspect_frames.py`, `docs/figures/cp1_frames.png`, `progress/cp1-A.md`.

## Tests
`tests/test_env.py`, 20 tests added (31 in the file):
- `MaxPoolSkip` on a scripted fake base env: full skip maxes the last two frames, sums 4 rewards, and takes `lives` from the last base step. Early termination on base step 1, 2, or 3 returns frame 1 or the max of the last two, with the partial reward sum, and stops stepping the base env.
- `MaxPoolSkip` on the real ALE vs 4 manual base steps, 300 random steps: reward sum, max-pooled obs, and lives are identical (and some steps score).
- Determinism: two `make_env(7)` with the same 200 random actions give identical obs, rewards, and lives.
- `noops` for seeds 0..19 equals the seed formula, is in [0, 30], and equals `ale.getEpisodeFrameNumber()` after reset (the no-ops really ran).
- Truncation at `max_episode_steps=5`; a real game over ends with `terminated=True`, a valid obs, and `lives == 0`.
- Transforms: shape and dtype for {84, 96} × {area, max} and `to_agent_obs`. `max` equals a numpy adaptive max-pool reference on random non-square input (this catches H/W swaps). A one-pixel object survives `max` but is dimmed by `area`. An unknown method raises.
- `AgentObsStack` drops the oldest of 5 pushes, and reset clears.
- Laser flicker (seeds 0 and 2, 100 steps): max-pooled frames show the laser at least as often as the single last raw frame.

`pytest -m "not slow and not gpu and not network"` → 46 passed in 13 s (macOS x86_64, Python 3.12.6, see Deviations). The `env` tests are part of that run. No `slow`, `network`, or `gpu` tests exist in this workflow.

## Results
`python scripts/inspect_frames.py --laser-stats` (seeds 0 and 2, 300 firing steps each, 110 frames with the laser between the aliens and shields):

| Version | Visible |
|---|---|
| raw max-pooled | 110/110 (100%) |
| raw last frame, no max-pool | 55/110 (50%): **100% for seed 0, 0% for seed 2** |
| 84 area RGB / grey | 72/110 (65.5%) |
| 84 max RGB / grey | 110/110 (100%) |
| 96 area RGB / grey | 72/110 (65.5%) |
| 96 max RGB / grey | 110/110 (100%) |

- **Laser calibration** (recorded in the script): the player laser and the alien bullets are the same colour, (142,142,142). The game draws the player laser only on **even** episode frame numbers (always moving up) and alien bullets only on **odd** ones (always moving down). This was checked on 4 seeds over full episodes. The script uses that parity to separate the laser from alien bullets.
- **Why the raw last frame is 0% or 100%:** each agent step is 4 frames, so the parity of the last frame equals the parity of the no-op count for the whole episode. With even no-ops (seed 0: 26) the last frame always has the laser; with odd (seed 2: 25) it never does. Without the max-pool, half the episodes would have no visible player laser at all.
- **Why area fails:** the laser is 1 raw pixel wide, and each 84-wide output column averages about 1.9 raw columns. When the laser falls inside one output column it keeps about 75/255 (visible); when it straddles two, each gets about 37 (< 40, invisible). Visibility therefore depends on the firing column, so it differs by seed (87% vs 44%).
- Figure: `docs/figures/cp1_frames.png`.

## Deviations
- **D10 trade-off to discuss:** by the roadmap rule the answer is 84/max. But the figure shows that 84/max turns the aliens into solid blocks (their shapes and animation frames are lost); 96/max keeps more shape. Nothing in the rule scores sprite detail. If the team cares about alien animation, 96/max is the alternative (it also passes at 100%).
- `inspect_frames.py` takes `--seeds/--steps/--out/--laser-stats` rather than `--config/--set`, like `play_real.py` in cp0: it has no configurable model or data parameters.
- `configs/data/*.yaml` is not created here (4.3 says D10 is recorded there). The collector config belongs to CP2-A, which should set `resolution: 84` and `resize_method: max`.
- ALE's own 108,000-frame cap (carried over from cp0): ale-py reports it as `truncated=True, terminated=False`, the same as our `TimeLimit`, so either way it maps to `end_reason = truncated`. No code change.
- **Local environment (not committed):** this laptop is an Intel Mac. PyTorch has no Intel-Mac wheels after 2.2.2, which doesn't support Python 3.13, so `setup.sh` cannot build a working venv here. I used a Python 3.12 venv with `torch==2.2.2`, `numpy<2`, `opencv-python<4.11`, `ale-py 0.10.2`, `gymnasium 1.0.0`, `stable-baselines3 2.4.1` (`pip check` clean), and `pip install -e . --ignore-requires-python`. The laser parity was verified on ale-py 0.10.2; the integrator should re-run `--laser-stats` on 0.12 to confirm.

## Interface change requests
None.

## Follow-ups
- Integrator: re-run `python scripts/inspect_frames.py --laser-stats` with the team's ale-py 0.12 and check the numbers match.
- CP2-A: use `to_model_frame(raw, 84, "max")` and record it in `configs/data/*.yaml`, unless the team picks 96 after discussing the trade-off above.
- Team: decide whether the setup should support Intel Macs (Python 3.12 + torch 2.2.2) or whether those machines only use the cluster.
