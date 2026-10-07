# cp0: Repo skeleton and frozen interfaces

## Summary
- The `wm/` package exists with the full layout from ROADMAP 4.1. Implemented in full: `wm/data/schema.py`, `wm/controls.py`, `wm/config.py`. `wm/env.py` is a working first version (seeded no-op reset, `MaxPoolSkip`, time limit, `to_model_frame("area")`, `to_agent_obs`, `AgentObsStack`).
- Every other `wm/` module is a stub. Modules with `[FROZEN]` interfaces have the signatures, and their bodies raise `NotImplementedError`. Modules without one only have a docstring naming the owning workflow.
- `tests/` has `conftest.py` (skips gpu tests without CUDA; sets the SDL dummy driver), `fixtures.py` (`TinyDenoiserNet` in full, `make_fake_shard` stub), and a fast suite that runs in about 2 s.
- Play scripts: `play.py` moved to `scripts/play_gym.py`. `scripts/play_real.py` is new, built on `wm.env` and `wm.controls`.
- `pyproject.toml`, plus `-e .` at the end of `requirements.txt`, so `python scripts/<x>.py` can import `wm`.

## Files
Created:
- `wm/__init__.py`, `wm/{agents,data,diffusion,models,train,eval}/__init__.py`
- `wm/config.py`, `wm/controls.py`, `wm/env.py`, `wm/inference.py`
- `wm/data/{schema,writer,reader,dataset,ram,coverage}.py`
- `wm/diffusion/{edm,sampler}.py`, `wm/models/{unet,embeddings,idm}.py`, `wm/train/trainer.py`
- `wm/eval/{metrics,baselines,visualize,rollout,plots,detectors,counterfactual}.py`
- `wm/agents/{policies,pretrained}.py`
- `scripts/play_real.py`
- `tests/{conftest,fixtures,test_smoke,test_controls,test_env,test_config,test_fixtures,test_play_real}.py`
- `pytest.ini`, `pyproject.toml`, `progress/TEMPLATE.md`, `progress/cp0.md`
- `configs/{data,model,train,eval}/.gitkeep`, `docs/figures/.gitkeep`

Modified or moved:
- `play.py` → `scripts/play_gym.py` (WideRender fix)
- `requirements.txt` (exact list from 4.2, plus `-e .` for the `wm` package)
- `.gitignore`, `README.md`

## Tests
Added:
- `test_smoke.py`: every `wm` module imports. A fresh interpreter importing all of `wm` does not load pygame.
- `test_controls.py`: all 8 key combinations. `ACTION_TO_MOVE_FIRE` agrees with `keys_to_action`. `FireLatch` behaviour on a tap, a hold, and idle.
- `test_env.py` (`env` marker, except the transform tests):
  - reset obs is (210,160,3) uint8, with `noops` in [0,30] and `seed == 0`
  - one step works
  - later resets use `seed + counter`
  - `noops` is derived from the seed
  - transform shapes and dtypes
  - `AgentObsStack` shift and zero-fill
- `test_config.py`:
  - nested override with numeric values, including `2e-4`
  - bools and lists
  - an unknown key and a malformed override both raise
  - `from_dict` builds nested dataclasses and rejects unknown keys
- `test_fixtures.py`: `TinyDenoiserNet` output shape for K ∈ {1,4} and C ∈ {1,3}, zero at init. With a non-zero output conv, every input changes the output.
- `test_play_real.py`: `display_array` widens ×2 horizontally and transposes without changing the env frame. The headless loop runs 10 ticks and exits (`env` marker).

Commands (Python 3.13.2, torch 2.14.1, gymnasium 1.4.0, ale-py 0.12.1, macOS arm64, CPU):
- `pytest -m "not slow and not gpu and not network"` → 26 passed in about 2 s
- `pytest -m "slow or network or env"` → 4 passed (only `env` tests exist so far)

Sanity check (not a test, since CP1-A owns it): two `make_env(7)` runs with the same 100 random actions gave identical frames.

**Not verified by me (no display): a human must run `python scripts/play_gym.py` and `python scripts/play_real.py` and confirm both are playable. This is the Phase 0 "done when" condition.**

## Results
None. Phase 0 has no numbers.

## Deviations
- **FireLatch semantics** (not in the spec):
  - `press()` is called on a Space key-down event and sets a counter to `n`. `tick(held)` is called once per agent step and returns `held or counter > 0`, then decrements the counter.
  - So a tap fires for the next `n` ticks (default 4), even if the key was released before the tick read it, and holding keeps firing.
- **WideRender in `play_gym.py`:** fixed rather than removed, by using `np.repeat(..., 2, axis=1)`. This matches the ×2 display widening in 4.3. An integer factor needs no cv2 call and gives even pixel columns, where nearest-neighbour at 1.5 does not.
- **`make_env` reset with an explicit seed:** `env.reset(seed=s)` restarts the episode counter from `s`. `reset()` with no seed uses `seed + episode_counter` as specified. 4.3 doesn't cover explicit seeds.
- **`AgentObsStack.reset/push` return the stacked (84,84,n) array.** The spec doesn't say what they return.
- **`wm/config.py`:**
  - `load_config(path, overrides)` returns a nested dict. An override key that doesn't exist in the YAML raises `KeyError`.
  - A separate `from_dict(cls, data)` builds a dataclass (nested dataclass fields supported) and raises `KeyError` on unknown keys.
  - Values are parsed with a `yaml.SafeLoader` subclass that also reads `2e-4` as a float. Plain PyYAML follows YAML 1.1 and returns the string `"2e-4"`, which would silently break `--set train.lr=2e-4`.
- **Config dataclass fields not fixed by the roadmap:**
  - `EDMConfig`: `sigma_data`, `p_mean`, `p_std`, `sigma_min`, `sigma_max`. The last two are the training σ clamp.
  - `SamplerConfig`: `num_steps`, `sigma_min`, `sigma_max`, `rho`, `method`.
  - `karras_sigmas(num_steps, sigma_min, sigma_max, rho)`.
  - `CopyLastFrame(K, C, H, W)`.
  - `make_dataloader(cfg: Any, split)`.
  - `UNetConfig` has no defaults for `C, K, H, ch`, because D10 (resolution) is still open.
  - Owning workflows may adjust any of these, since none are `[FROZEN]`.
- **`wm/data/schema.py`:**
  - `ROW_FIELDS` maps name → (dtype, trailing shape). For `frames` the shape is `None`, since it comes from the writer's `frame_shape`.
  - `EPISODE_FIELDS` maps name → dtype.
  - The file also holds the other 4.4 constants (`FILE_ATTRS`, chunking, compression, flush size, `MAX_ROWS`, `EPISODE_UID_STRIDE`) and the seed bases `EVAL_SEED_BASE` (CP2-A) and `HUMAN_SEED_BASE` (CP3-A).
- **`.gitignore` uses `/data/` and `/runs/`, anchored to the repo root.** An unanchored `data/` also ignored `wm/data/` and `configs/data/`. Also added: `*.egg-info/`.
- **Added `pyproject.toml` and an `-e .` line at the end of `requirements.txt`.** `setup.sh` is unchanged. Without them `python scripts/play_real.py` cannot import `wm`, because Python puts `scripts/` on `sys.path`, not the repo root. Dependencies stay in `requirements.txt`. `pytest.ini` also sets `pythonpath = .`, so tests work without the install.
- **`play_real.py` takes `--fps`, `--seed`, and `--scale`, not `--config`/`--set`.** It has no configurable parameters beyond these.

## Interface change requests
None.

## Follow-ups
- **SDL2 duplicate on macOS:** opencv-python 5.0 bundles its own `libSDL2`, and so do the headless and 4.x wheels. `play_real.py` loads it alongside pygame's copy, through `wm.env` → cv2. objc warns "Class SDLApplication is implemented in both … may cause spurious casting failures and mysterious crashes". The headless test runs cleanly. The human playtest should watch for crashes. If one shows up, the likely fix is to load cv2 lazily, so it isn't imported for display-only scripts.
- **For CP1-A:**
  - ALE v5 also caps an episode at `max_num_frames_per_episode=108000` emulator frames, which is the same as 27,000 agent steps. Reset no-ops count toward that cap, so the ALE can truncate up to 8 agent steps before our `TimeLimit` does. Either way the episode ends with `truncated=True`. Decide whether this matters for `end_reason`.
  - `to_model_frame(method="max")` still raises `NotImplementedError`.
- **Local venv:** the previous `.venv` was Python 3.14. I recreated it with `setup.sh` (3.13). torch 2.14.1 came in as a transitive dependency (via lpips and SB3), so CPU torch works out of the box on laptops. The cluster still needs the CUDA build.
