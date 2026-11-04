# ROADMAP: Neural World Model for Space Invaders

An action-conditioned diffusion model that learns to *be* Atari Space Invaders: given the
last K frames and the player's actions, it generates the next frame. Run in a loop, it is a
playable game with no game code. This is a simplified, class-scale version of GameNGen
(Valevski et al., 2024, "Diffusion Models Are Real-Time Game Engines"), built in pixel space
in the style of DIAMOND (Alonso et al., 2024).

**Read the whole file before writing any code.** Section 4 is the technical source of truth.
Section 5 lists the phases. Section 1 tracks progress.

---

## 0. How to use this document

**For Claude Code (implementing a workflow):**
1. Read sections 2, 3, and 4 completely, then the section for your workflow in section 5.
2. Work only on your workflow's branch, and only in the files your workflow owns
   (listed under "Owns"). Never change a `[FROZEN]` interface in section 4. If you think one
   must change, stop and write the request under "Interface change requests" in your
   progress note (section 3.5).
3. Write the tests listed for your workflow first, then the implementation. Run
   `pytest -m "not slow and not gpu and not network"` before every commit. It must pass.
4. When the workflow is done, fill in `progress/cpN-X.md` (template in section 3.5). Do
   **not** edit section 1 of this file on a workflow branch. The integrator updates it at
   the checkpoint.
5. If anything in this file is ambiguous, choose the simplest option consistent with
   section 4, and record the choice under "Deviations" in your progress note.

**For the planning chat (generating prompts):** the team will say which checkpoint and
workflow they are on (e.g. "CP2-B"). Produce Claude Code prompts using the templates in
section 6: one prompt per workflow, plus one for the checkpoint integration. Use section 1
to see what already exists.

**Status values used below:** `TODO`, `IN PROGRESS`, `DONE`, `BLOCKED`.

---

## 1. Progress

### 1.1 Status board

| Checkpoint | Workflow | Branch | Owner | Status |
|---|---|---|---|---|
| Pre | Environment boots, game playable | `main` | all | DONE |
| P0 | Repo skeleton and frozen interfaces | `cp0/skeleton` | Syed | DONE |
| CP1 | A: Environment module and resolution decision | `cp1/A-env` | Aunonto | DONE |
| CP1 | B: HDF5 storage (writer, reader, validator) | `cp1/B-storage` | Syed | DONE |
| CP1 | C: Policies (pretrained agent, random, epsilon mix) | `cp1/C-policies` | Michael | DONE |
| CP2 | A: Collector, debug and eval datasets, coverage report | `cp2/A-collect` | Aunonto | DONE |
| CP2 | B: PyTorch window dataset and loader | `cp2/B-dataset` | Syed | DONE |
| CP2 | C: EDM core and sampler | `cp2/C-edm` | Michael | DONE |
| CP3 | A: Metrics, baselines, visualization, human recorder | `cp3/A-eval` | Aunonto | TODO |
| CP3 | B: Trainer | `cp3/B-trainer` | Syed | TODO |
| CP3 | C: U-Net denoiser | `cp3/C-unet` | Michael | TODO |
| CP4 | A: Autoregressive rollouts and horizon evaluation | `cp4/A-rollout` | Aunonto | TODO |
| CP4 | B: Context noise augmentation runs | `cp4/B-noiseaug` | Syed | TODO |
| CP4 | C: Inference engine and sampler benchmarks | `cp4/C-inference` | Michael | TODO |
| CP5 | A: Controllability evaluation and inverse dynamics | `cp5/A-control` | Aunonto | TODO |
| CP5 | B: Full dataset and final training run | `cp5/B-fullrun` | Syed | TODO |
| CP5 | C: Action conditioning and classifier-free guidance | `cp5/C-cfg` | Michael | TODO |
| CP6 | A: Final evaluation and figures | `cp6/A-final-eval` | Aunonto | TODO |
| CP6 | B: Playable demo | `cp6/B-demo` | Syed | TODO |
| CP6 | C: Ablations | `cp6/C-ablations` | Michael | TODO |
| CP7 | Stretch: agent trained inside the world model | `cp7/*` | any | TODO |

### 1.2 Done so far

**Phase 0 (tag `cp0`, details in `progress/cp0.md`):**

- `wm/` package with the full layout from 4.1, installed in editable mode (`pyproject.toml`
  plus `-e .` at the end of `requirements.txt`) so scripts can `import wm`.
- Implemented in full: `wm/data/schema.py`, `wm/controls.py` (`keys_to_action`,
  `ACTION_TO_MOVE_FIRE`, `FireLatch`), `wm/config.py` (`load_config` with `--set`
  overrides, `from_dict`; `2e-4`-style floats parse correctly).
- `wm/env.py` first working version: seeded no-op reset, `MaxPoolSkip`, time limit,
  `to_model_frame("area")`, `to_agent_obs`, `AgentObsStack`. `method="max"` still raises
  (CP1-A).
- Every other `wm/` module is a stub. `[FROZEN]` signatures exist and raise
  `NotImplementedError`. Config dataclass fields that the roadmap doesn't fix are listed in
  the progress note; owning workflows may change them.
- `scripts/play_gym.py` (the old `play.py`; `WideRender` now widens ×2) and
  `scripts/play_real.py` (custom pygame loop on `wm.env`: max-pooled frames, `FireLatch`,
  `--fps/--seed/--scale`, ×2 display widening, R resets, Esc quits).
- `tests/`: `conftest.py` (SDL dummy driver, gpu auto-skip), `fixtures.py`
  (`TinyDenoiserNet` in full, `make_fake_shard` stub for CP1-B), and a fast suite of 26
  tests.
- `.gitignore` anchors `/data/` and `/runs/` to the repo root (unanchored, they also
  ignored `wm/data/` and `configs/data/`).

**CP1 (tag `cp1`, details in `progress/cp1-A.md`, `cp1-B.md`, `cp1-C.md`):**

- `wm/env.py`: `to_model_frame(method="max")` added (torch imported lazily). `MaxPoolSkip`
  and `make_env` already matched 4.3 and are now pinned down by 31 tests in `test_env.py`.
  The ALE's own 108,000-frame cap reports `truncated=True`, so it maps to
  `end_reason = truncated` like our `TimeLimit` (resolves the cp0 carry-over).
- `scripts/inspect_frames.py`: `docs/figures/cp1_frames.png` and `--laser-stats`. The
  player laser and alien bullets share colour (142,142,142); the laser is drawn only on even
  emulator frames, bullets only on odd ones. Without the max-pool, the laser is invisible
  for the whole of every episode with an odd no-op count.
- `wm/data/writer.py` (`ShardWriter`), `wm/data/reader.py` (`list_shards`, lazy picklable
  `ShardReader`), `scripts/validate_data.py` (all 4.4 invariants plus statistics), and
  `make_fake_shard` in full with `encode_frame`/`decode_frame` helpers in `tests/fixtures.py`.
  Writer behaviour not fixed by 4.4: `close()` with an open episode warns and discards it;
  shards open with `"w-"` (never overwrite); caller must pass all non-writer attrs.
- `wm/agents/`: `RandomPolicy`, `EpsilonMixPolicy`, `ZooAgentPolicy`
  (`sb3/ppo-SpaceInvadersNoFrameskip-v4`, obs `(4,84,84)`; a legacy-`gym` unpickling shim,
  no `gym` dependency). `scripts/eval_agent.py` prints the score table.
- `scripts/smoke_cp1.py` (integration): ε=0.2 agent, 2 episodes capped at 500 steps, 84/max
  shard plus a 4× frame grid. `validate_data.py` passes.

**CP2 (tag `cp2`, details in `progress/cp2-A.md`, `cp2-B.md`, `cp2-C.md`):**

- `scripts/collect.py`: spawn multiprocess collector driven by `configs/data/{debug,eval}.yaml`
  (84/max). One `ShardWriter` per (job, worker) part (D18), per-episode ε, `_DONE` markers
  and resume, `summary.json`. Replay determinism is tested (D9).
- `wm/data/ram.py`: verified RAM map (invaders left 17, lives 73, score BCD 102/104, player
  x 28, alien block x 26 / y 16, UFO 76 ≥ 4). AtariARI `player_score` and `enemies_y` failed
  and were replaced; `missiles_y` was dropped. Figure: `docs/figures/cp2_ram_verify.png`.
- `wm/data/coverage.py` and `scripts/coverage_report.py`: report and montages in
  `docs/figures/cp2_coverage_*`. 4 of 5 targets met; wave transitions revised (D17).
- `wm/data/dataset.py`: `WindowDataset` (4.5) reading one contiguous slice per window,
  optional `in_memory`, `DataConfig`, `make_dataloader`. `scripts/bench_loader.py`.
- `wm/diffusion/edm.py` (`EDMDenoiser`, `loss_terms`, `loss_by_sigma`) and
  `wm/diffusion/sampler.py` (`karras_sigmas`, Euler/Heun `sample`, batched action CFG that
  takes the single-pass path at `cfg_scale=1`). `configs/train/edm_default.yaml`.
- `scripts/smoke_cp2.py` (integration): `make_dataloader` over `data/debug/` into
  `EDMDenoiser(TinyDenoiserNet)`, 200 CPU steps, checks finite and decreasing loss.

**Carried over to CP3:**

- CP3-B: call `per, sigma = edm.loss_terms(...)`, `loss = per.mean()`, and log
  `loss_by_sigma(per, sigma)`. Pass `WindowDataset` batches as they are (5-D context is
  flattened inside). Action dropout and context noise are the trainer's job. For sample
  grids, call `sample(ema_denoiser.eval(), ...)`; `sample` only applies `no_grad`. Copy the
  `edm:` and `sampler:` blocks from `configs/train/edm_default.yaml` into `debug.yaml`.
- CP3-B: exact resume needs the global torch RNG restored (σ and ε are drawn from it).
  Use the lazy loader with 4–8 workers (3.8k–6k samples/s on an M-series laptop);
  `in_memory` on `data/debug/` needs ~10.6 GB per split. Build each loader once per run
  (`persistent_workers`). Nest `DataConfig` in the train config or duck-type its fields.
- CP3-B: `data/debug/` and `data/eval/` are not in git. Regenerate with
  `python scripts/collect.py --config configs/data/{debug,eval}.yaml --workers {4,2}`
  (keep `--workers` the same as the original run so writer ids and seeds match), then copy
  to Discovery scratch. Agent actions can differ across torch builds, so datasets from
  different machines are equally valid but not byte-identical.
- CP3-A: `data/eval/` episodes all have ≥ 262 rows, enough for K + 200 horizon.
- CP4-C (D11): on Gaussian data, Euler at N=3 gives ≈ 0.53 of the true std (expect blur
  on real frames); Heun at N=3 overshoots 3.75× before the clamp. Check both in the benchmark.
- CP5-B: collection laptop (Aunonto) has little disk left; collect `data/full/` elsewhere.
  `to_model_frame(method="max")` costs ~1.6 ms per frame (~20% of collection time); a numpy
  version would speed up CP5 collection (CP1-A-owned file, change needs the integrator).
- CP5-C: the CFG hook is `_guided_denoise` in `wm/diffusion/sampler.py`.
- On macOS, opencv bundles its own `libSDL2` next to pygame's (objc duplicate-class
  warning). No crash seen yet; if one appears, import cv2 lazily in `wm/env.py`.
- Keep the repo outside iCloud-synced folders. iCloud sets `UF_HIDDEN` on `.venv` files and
  Python 3.13 then skips the editable-install `.pth`, so `import wm` fails (see README).

### 1.3 Decisions log

| # | Decision | Status |
|---|---|---|
| D1 | Game: `ALE/SpaceInvaders-v5`, minimal action set of 6 actions | Final |
| D2 | Base env `frameskip=1`; our own 4-frame skip with max over the last 2 frames; sticky actions off | Final |
| D3 | `.venv`, no Docker (Discovery uses a venv too; Apptainer is the cluster option if a container is ever needed) | Final |
| D4 | Data uses raw square-pixel frames. Aspect correction is only for display. | Final |
| D5 | Default game mode and difficulty only; the one-shot rule stays. Demo difficulty is eased with starting scenarios and slower speed. | Final |
| D6 | Final model RGB (following DIAMOND). Greyscale for debugging is derived on the fly from stored RGB. | Final |
| D7 | Pixel-space diffusion (no VAE), unlike GameNGen. 84×84 frames are small enough. | Proposed in roadmap |
| D8 | Store RAM (128 bytes) per step for coverage stats and detector calibration. Never used as model input. Verified map in `wm/data/ram.py` (CP2-A). | Final (CP2-A) |
| D9 | Ground truth for rollouts comes from deterministic replay (seed, no-op count, actions). Replay reproduces every stored frame, reward, life count and RAM byte exactly (CP2-A test). | Final (CP2-A) |
| D10 | 84×84, `method="max"`. `area` keeps the laser in only 65.5% of frames at 84 and 96; `max` keeps 100% at both. 96/max keeps more alien shape (84/max turns aliens into blocks) and stays an option if sprite detail matters. | Final (CP1-A) |
| D11 | Default sampling steps and classifier-free guidance weight | Open, decided in CP4-C and CP5-C |
| D12 | Training runs on Dartmouth's Discovery cluster (Slurm), with checkpoint and auto-resume so jobs survive time limits and queue waits | Final |
| D13 | Track owners: A = Aunonto, B = Syed, C = Michael. Syed takes B because he is the only one with Discovery access (a change from the proposal's split). | Final |
| D14 | Python 3.13 for the venv (pygame has no 3.14 wheels on macOS), enforced by `setup.sh` | Final |
| D15 | `wm` is installed as an editable package (`pyproject.toml`, `-e .` in `requirements.txt`); dependencies stay in `requirements.txt` | Final |
| D16 | Pretrained agent: `sb3/ppo-SpaceInvadersNoFrameskip-v4` (PPO, CPU, `deterministic=False`). Scores over 5 seeds: agent 1041 ± 264, ε=0.2 596 ± 137, ε=0.3 291 ± 180, random 139 ± 62. | Final (CP1-C) |
| D17 | Wave-transition coverage target: ≥ 20 for the 500k debug set (24 measured; random play never clears a wave), ≥ 50 kept for `data/full/`. Add an ε ≈ 0.05 agent job in CP5 if more late-wave frames are needed. | Final (CP2-A) |
| D18 | Collector writer ids are `w = job_index * workers + worker`, so episode seed = `base_seed + episode_uid`. Resuming with a different `--workers` is refused. | Final (CP2-A) |

### 1.4 Open questions

- Discovery GPU partition, GPU type, memory, and the job time limit. Fill in here: ____
- Does Discovery have a Python 3.13 module? `setup.sh` requires `python3.13`. If not, the
  cluster setup needs its own path (pygame is only needed on laptops).
- How much human play to record (target: at least 1 hour total, collected during CP3 to CP5).
- Intel Macs: PyTorch has no Intel-Mac wheels that support Python 3.13 (CP1-A used 3.12 +
  torch 2.2.2 + `numpy<2`). Support that setup, or use those machines only with the cluster?

### 1.5 Results log

The integrator adds rows at each checkpoint.

| Date | Checkpoint | Result |
|---|---|---|
| 2026-10-07 | P0 | Fast suite 26 passed; `env` tests 4 passed (macOS arm64, CPU, Python 3.13.2). Same seed and actions give identical frames. |
| 2026-10-16 | CP1 | Fast suite 85 passed; `slow or network or env` 13 passed (macOS arm64, CPU, Python 3.13.2, ale-py 0.12.1). Laser visibility: 84/max and 96/max 100%, 84/area and 96/area 65.5%, raw single frame 50% (re-run on ale-py 0.12.1 matches CP1-A's 0.10.2 numbers). D10 = 84/max. |
| 2026-10-16 | CP1 | Agent scores (5 seeds): agent 1041 ± 264, ε=0.2 596 ± 137, ε=0.3 291 ± 180, random 139 ± 62. Writer 54,886 rows/s at 84×84 (fake frames). |
| 2026-10-16 | CP1 | `smoke_cp1.py`: 2 episodes × 501 rows at 84/max, 1,031 rows/s end to end (env + agent bound), 1.73 MiB, compression ratio 11.7. `validate_data.py` OK. 20-frame visual check: laser visible, frames square, not aspect-corrected. |
| 2026-11-04 | CP2 | Fast suite 165 passed after merging A, B, C (105 → 131 → 165; macOS arm64, CPU, Python 3.13.2). |
| 2026-11-04 | CP2 | `data/debug/` (integrator's regeneration, `--workers 4`, 2 min 42 s on 10 cores): 500,000 steps, 688 episodes (457 agent, 231 random), 500,688 rows, 740 MiB, compression 13.7, validator OK. Aunonto's original: 687 episodes, 741 MiB. `data/eval/`: 100 episodes (70 agent, 30 random), 39,860 rows, 68 MiB, seeds ≥ 1e9, validator OK. |
| 2026-11-04 | CP2 | Coverage on Aunonto's `data/debug/` (`docs/figures/cp2_coverage_report.md`): ≤ 6 invaders 8.48% (target 5%), wave transitions 24 (target 50, revised to 20, D17), game overs 675 (200), UFO visible 17.51% (1%), shields ≤ 50% 15.26% (10%). Life losses 1,958. UFO pixel detector agrees with RAM on 99.93%. Mean score: agent 463.8, random 148.2. |
| 2026-11-04 | CP2 | `smoke_cp2.py` on `data/debug/` (471,846 train windows, K=4, batch 32, 4 workers, `TinyDenoiserNet(hidden=32)`, AdamW 2e-3): 200 CPU steps in 59 s, no NaN, mean loss first 20 steps 1.037, last 20 steps 0.394. |
| 2026-11-04 | CP2 | `bench_loader.py` on `data/debug/` (K=4, batch 64), samples/s for 0/2/4/8 workers: lazy 1,091 / 2,137 / 3,955 / 5,529; in memory 7,026 / 9,100 / 14,037 / 10,605 (12.7 s startup). Target ≥ 2,000 met with ≥ 2 lazy workers. |
| 2026-11-04 | CP2 | EDM Gaussian oracle (s = 0.3): Heun N=50 std/s 1.003; Euler N=3 0.535 (closed form 0.532). Default N=3 schedule σ = [5.0, 0.283, 0.002, 0]. |

---

## 2. Project summary and relationship to GameNGen

| Aspect | GameNGen (DOOM) | This project (Space Invaders) |
|---|---|---|
| Data | RL agent trained to play like a human; its episodes are recorded | Pretrained RL Baselines3 Zoo agent with 20–30% random actions, plus random and human play |
| Space | Latent (Stable Diffusion v1.4 VAE); decoder fine-tuned for small details | **Pixel space**, no VAE (frames are 84×84 or 96×96) |
| Denoiser | Pretrained SD U-Net, fine-tuned | Small U-Net trained from scratch (10–40M parameters) |
| Diffusion | DDPM/DDIM style, 4 sampling steps | EDM (Karras et al., 2022), 3–10 Euler steps |
| Frame conditioning | Past frame latents concatenated on the channel axis | Past frames concatenated on the channel axis |
| Action conditioning | Learned action embeddings replace the text cross-attention | Action embeddings plus noise-level embedding, injected by adaptive group norm (AdaGN) |
| Context length | 64 frames | 4 frames by default (ablate 1, 4, 8) |
| Anti-drift | Gaussian noise added to context frames during training; noise level discretized and given to the model | Same idea (section 4.8), with 10 levels up to σ=0.7 |
| Guidance | Classifier-free guidance on past-frame conditioning | Classifier-free guidance on **actions** (action dropout 10%) |
| Evaluation | PSNR and LPIPS against ground truth, FVD, human raters | PSNR, SSIM, LPIPS vs rollout horizon; controllability; ablations |

The main result of the project is the curve of **frame quality vs rollout horizon**,
compared with a copy-last-frame baseline and with and without context noise augmentation.

---

## 3. Team workflow and git

### 3.1 Tracks

Three tracks, kept roughly stable so each person builds up context. Owners follow the
proposal's work split, except that Syed and Michael swapped: Syed has the only Discovery
account, so he owns the track that runs the training jobs.

- **A: Environment and evaluation (Aunonto).** env module, collection, metrics, rollouts, controllability, final evaluation.
- **B: Data and training infrastructure (Syed).** storage, dataset, trainer, noise augmentation runs, full training run, demo.
- **C: Diffusion model (Michael).** EDM, sampler, U-Net, inference engine, CFG, ablations.

### 3.2 Branches and checkpoints

- `main` only receives merges at checkpoints. Tag each checkpoint `cp0`, `cp1`, and so on.
- At the start of checkpoint N, each track branches from tag `cp(N-1)`:
  `cpN/A-<name>`, `cpN/B-<name>`, `cpN/C-<name>`.
- The three workflows in a checkpoint run **in parallel** and must not depend on each
  other's unmerged code. Where a workflow needs something another workflow is building in
  the same checkpoint, it codes against the `[FROZEN]` interface in section 4 and uses the
  stub or fixture named in its section.
- Commit message prefix: `[cpN-X] short description`.
- Never commit data, checkpoints, or run outputs. `data/`, `runs/`, `*.h5`, `*.pt`, and
  `.venv/` are in `.gitignore`.

### 3.3 Checkpoint ritual (integrator)

One person acts as integrator for the checkpoint, rotating A → B → C.

1. Each workflow branch passes `pytest -m "not slow and not gpu and not network"` and has its progress note filled in.
2. Merge into `main` in the order **A, then B, then C** (unless the checkpoint section says otherwise). Run the full fast test suite after each merge.
3. Run the checkpoint's **integration checks** (listed at the end of each checkpoint).
4. Update section 1 of this file: status board, results log, decisions log. Remove resolved open questions.
5. Tag `cpN` and push. The next checkpoint branches from this tag.

### 3.4 File ownership

Each workflow lists the files it owns. Shared files have these rules:

- `requirements.txt`: append new lines only; never reorder or remove lines. The integrator resolves conflicts.
- `configs/`: each workflow adds new files; don't edit another workflow's config files.
- `tests/conftest.py` and `tests/fixtures.py`: owned by Phase 0. Workflows may **add** new fixtures at the end of `tests/fixtures.py`.
- `ROADMAP.md`: only the integrator edits it, and only at checkpoints.

### 3.5 Progress notes

Each workflow writes `progress/cpN-X.md`:

```markdown
# cpN-X: <workflow name>
## Summary            (what now exists, 3-6 lines)
## Files              (created / modified)
## Tests              (added; command run; result)
## Results            (numbers, figure paths)
## Deviations         (anything different from ROADMAP.md, and why)
## Interface change requests   (none / description)
## Follow-ups         (known issues, TODOs for later checkpoints)
```

### 3.6 Interface change protocol

If a `[FROZEN]` interface must change: write the request in the progress note, implement
the change in a backwards-compatible way if possible, and flag it to the integrator. The
integrator updates section 4 at the checkpoint, and the other tracks adapt at the start of
the next checkpoint.

---

## 4. Global technical specification (source of truth)

### 4.1 Repository layout

```
ROADMAP.md  README.md  requirements.txt  setup.sh  pyproject.toml  pytest.ini  .gitignore
wm/                              # the python package ("world model")
  __init__.py
  config.py                      # dataclass + YAML config loading
  controls.py                    # keyboard -> action mapping shared by play/record/demo
  env.py                         # make_env, MaxPoolSkip, frame transforms
  inference.py                   # WorldModel (inference API)
  agents/  policies.py  pretrained.py
  data/    schema.py  writer.py  reader.py  dataset.py  ram.py  coverage.py
  diffusion/  edm.py  sampler.py
  models/  embeddings.py  unet.py  idm.py
  train/   trainer.py
  eval/    metrics.py  baselines.py  visualize.py  rollout.py  plots.py  detectors.py  counterfactual.py
scripts/
  play_gym.py  play_real.py  play_record.py  play_model.py
  inspect_frames.py  eval_agent.py  collect.py  validate_data.py  coverage_report.py
  bench_loader.py  train.py  eval_single_step.py  eval_rollouts.py  bench_sampling.py
  eval_counterfactual.py  train_idm.py  make_scenarios.py  run_ablations.py
configs/  data/  model/  train/  eval/
tests/    conftest.py  fixtures.py  test_*.py
progress/ cpN-X.md
docs/     figures/  results.md
data/     (gitignored)  runs/  (gitignored)
```

### 4.2 Dependencies

`requirements.txt` after Phase 0 (append only afterwards):

```
gymnasium[atari]>=1.0.0
ale-py>=0.9.0
pygame>=2.5
opencv-python>=4.8
matplotlib>=3.7
numpy>=1.26
h5py>=3.9
pyyaml>=6.0
tqdm>=4.66
pytest>=8.0
stable-baselines3>=2.3
huggingface_sb3>=3.0
torchmetrics>=1.3
scikit-image>=0.22
lpips>=0.1.4
imageio[ffmpeg]>=2.34
tensorboard>=2.15
-e .                     # installs the wm package (pyproject.toml)
```

**PyTorch is installed separately**, using the command for the machine's CUDA version from
pytorch.org (`torch>=2.2`). Note this in the README. Code must run on CPU (slowly) when no
GPU is present. Choose the device with `"cuda" if torch.cuda.is_available() else "cpu"`
(and `"mps"` for local Mac debugging only).

### 4.3 Environment and preprocessing `[FROZEN]`

**Base env.** Every env in the project, including collection, evaluation, and the human
recorder, is created through `wm.env.make_env`:

```python
gym.register_envs(ale_py)
base = gym.make("ALE/SpaceInvaders-v5", frameskip=1, repeat_action_probability=0.0,
                full_action_space=False)          # obs_type="rgb" (default), obs (210,160,3) uint8
```

**Actions** (`wm/data/schema.py`):
`ACTION_MEANINGS = ["NOOP","FIRE","RIGHT","LEFT","RIGHTFIRE","LEFTFIRE"]`, `NUM_ACTIONS = 6`,
`NULL_ACTION = 6` (classifier-free guidance token; never appears in data). Assert
`base.unwrapped.get_action_meanings() == ACTION_MEANINGS` in `make_env`.

**`MaxPoolSkip(gym.Wrapper)`, `skip=4`.** `step(a)` repeats `a` on the base env 4 times,
sums the rewards, and returns `obs = np.maximum(frame[-2], frame[-1])` (uint8, 210×160×3)
from the last two raw frames. If the episode terminates or truncates early, it stops and
max-pools over the frames it has (one frame means that frame is returned). `info["lives"]`
comes from the last base step. This is the same scheme as `AtariPreprocessing` and SB3's
`MaxAndSkipEnv`. The max-pool is what keeps Space Invaders' alternating-frame sprites, the
lasers in particular, visible.

**`make_env(seed: int, noop_max: int = 30, max_episode_steps: int = 27_000) -> gym.Env`.**
- `reset()` calls `base.reset(seed=seed)`, then performs `n_noop` base-env NOOP steps,
  with `n_noop = np.random.default_rng(seed).integers(0, noop_max + 1)`. The returned obs is
  the max of the last two raw frames, or the reset frame if `n_noop < 2`.
  `info["noops"] = n_noop`, `info["seed"] = seed`. Later resets of the same env object use
  `seed + episode_counter`, which is also reported in `info["seed"]`.
- One agent step = 4 emulator frames. 15 agent steps = 1 second of real game time.
- Truncate after `max_episode_steps` agent steps.
- No terminal-on-life-loss. Life loss is part of what the model must learn.
- **Determinism requirement:** same seed + same action sequence ⇒ bit-identical
  observations. This is tested in CP1-A and relied on in CP2 and CP5.

**Frame transforms** (all inputs and outputs uint8, channels last):
- `to_model_frame(raw, size: int, method: str = "area") -> (size,size,3)`
  - `"area"`: `cv2.resize(raw, (size, size), interpolation=cv2.INTER_AREA)`. cv2 takes the size as **(W, H)**.
  - `"max"`: `torch.nn.functional.adaptive_max_pool2d` to (size, size), applied per channel. This keeps one-pixel bright objects on the black background.
  - Default `size` and `method` are chosen in CP1-A and recorded in D10 and in `configs/data/*.yaml`.
- `to_agent_obs(raw) -> (84,84)`: `cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)`, then
  `cv2.resize(..., (84,84), interpolation=cv2.INTER_AREA)`. This matches SB3's `WarpFrame`.
- `AgentObsStack(n=4)`: holds the last 4 agent obs, channels last (84,84,4). `reset(obs)`
  fills with zeros and then pushes `obs`, matching SB3 `VecFrameStack`. `push(obs)` drops the oldest.

**Pixel aspect.** Frames are always square-pixel and squashed to size×size. Any aspect
widening (e.g. ×2 horizontally) is done only when drawing to a screen or a video.

### 4.4 Data storage schema `[FROZEN]` (`schema_version = 1`)

- One HDF5 file per shard: `data/<dataset_name>/shard_w<worker:03d>_<index:05d>.h5`.
  Each collector worker writes only its own shards (no concurrent writers).
- Rows are frames in time order, grouped by **episode**. Episodes never span shards: a
  writer rotates to a new shard only between episodes, once it has at least `max_rows` (default 100,000).
- An episode with T agent steps has **T+1 rows** (first frame after no-ops through the final frame).

Per-row datasets (N rows; created resizable with `maxshape=(None, ...)`):

| Name | dtype / shape | Meaning |
|---|---|---|
| `frames` | uint8 (N,H,W,3), `chunks=(32,H,W,3)`, `compression="lzf"` | model frame (`to_model_frame`) |
| `actions` | int8 (N,) | action taken **at** this frame (it leads to the next row); `-1` on the last row of each episode |
| `rewards` | float32 (N,) | reward from that action (sum over 4 emulator frames); `0` on the last row |
| `lives` | int8 (N,) | `info["lives"]` at this frame |
| `ram` | uint8 (N,128) | `env.unwrapped.ale.getRAM()` at this frame |
| `episode_idx` | int32 (N,) | shard-local episode index (row index into `episodes/*`) |
| `step_in_episode` | int32 (N,) | 0 for the first row of an episode |
| `terminal` | bool (N,) | True only on the last row of an episode that ended with `terminated=True` |

Per-episode datasets in group `episodes/`, all (E,): `start` int64 (first row),
`length` int64 (rows = T+1), `episode_uid` int64 (globally unique:
`worker * 10_000_000 + local_counter`), `seed` int64, `noops` int16, `source` int8
(`0=agent, 1=random, 2=human`, constants in `schema.py`), `epsilon` float32 (-1 if not
applicable), `end_reason` int8 (`1=terminated, 2=truncated, 3=stopped` (human quit)),
`return_` float32.

File attributes: `schema_version`, `env_id`, `frameskip=4`, `maxpool=True`, `resolution`,
`resize_method`, `ale_py_version`, `gymnasium_version`, `git_commit`, `created_utc`,
`collector_config` (JSON string).

**Invariants** (checked by `scripts/validate_data.py`): the episodes cover rows `0..N-1`
contiguously with no gaps or overlap; `actions == -1` exactly on each episode's last row
and values `0..5` everywhere else; `terminal` only on last rows; `episode_idx` and
`step_in_episode` agree with `episodes/`; dtypes and shapes match the table.

**Writer API** (`wm/data/writer.py`):

```python
w = ShardWriter(out_dir, worker: int, frame_shape=(H,W,3), attrs: dict, max_rows=100_000)
w.begin_episode(frame, ram, lives, seed, noops, source, epsilon)   # may rotate shard first
w.add_step(action, reward, next_frame, next_ram, next_lives)       # buffered; flush every 1024 rows
w.end_episode(end_reason, terminated: bool)                        # sets last row action=-1, terminal
w.close()
```

**Reader API** (`wm/data/reader.py`): `list_shards(dataset_dir) -> list[Path]`;
`ShardReader(path)` opens lazily on first access (needed for DataLoader workers) and
provides `num_rows`, `attrs`, `episodes` (dict of numpy arrays), `episode_rows(e) -> slice`,
and `read(name, rows) -> np.ndarray` for any per-row dataset with a slice or sorted index array.

**Datasets:**
- `data/debug/`: ~500k agent steps (CP2).
- `data/eval/`: held-out trajectories for rollouts, with **seeds ≥ 1,000,000,000** so they never overlap training seeds (CP2).
- `data/human/`: human play (CP3 onwards).
- `data/full/`: 2–5M steps (CP5).

### 4.5 Training sample contract `[FROZEN]`

`WindowDataset(dataset_dirs, K, split, grayscale=False, val_fraction=0.05, seed=0, *, in_memory=False)`:

- `in_memory=True` loads each shard's frames and actions into shared-memory tensors; items are identical to the lazy path. `seed` does not affect the split; `make_dataloader(cfg: DataConfig, split)` uses `cfg.seed` for shuffling and worker seeds. `DataConfig` lives in `wm/data/dataset.py`.
- `shard` in each item indexes the global list of shards from all `dataset_dirs`, sorted by full path.

- Valid **target rows** `i` are rows with `i - K >= start` of the same episode (all of
  `i-K..i` are in one episode). Context rows are `i-K..i-1`, context actions are
  `actions[i-K..i-1]` (none are -1, because the last row of an episode is only ever a
  target), and the target is row `i`. The number of windows per episode is `max(0, length - K)`.
- **Split by episode, never by frame:**
  `val = (zlib.crc32(f"{shard_name}:{episode_uid}".encode()) % 10_000) < val_fraction * 10_000`.
- Each item is a dict:
  - `context`: float32 (K,C,H,W) in [-1,1], oldest first
  - `actions`: int64 (K,); `actions[-1]` is the action that produced the target
  - `target`: float32 (C,H,W) in [-1,1]
  - `row`: int64, `shard`: int64
- Normalization: `x = uint8 / 127.5 - 1`. Greyscale (C=1) uses the luma weights
  `0.299 R + 0.587 G + 0.114 B`, applied to uint8 before normalizing.
- The model sees the context flattened to (K·C, H, W).

### 4.6 EDM formulation (`wm/diffusion/edm.py`)

Notation: clean target `y`, noise level `σ`, noisy input `x = y + σ·ε`, `ε ~ N(0, I)`.

Preconditioning (Karras et al., 2022), with `σ_d = 0.5`:
- `c_skip = σ_d² / (σ² + σ_d²)`
- `c_out = σ·σ_d / sqrt(σ² + σ_d²)`
- `c_in = 1 / sqrt(σ² + σ_d²)`
- `c_noise = ln(σ) / 4`
- `D(x; σ, cond) = c_skip·x + c_out·F(c_in·x, c_noise, cond)`

Training:
- Sample `ln σ ~ N(P_mean, P_std²)` with `P_mean = -0.4`, `P_std = 1.2`, then clamp σ to `[2e-3, 20]`. These are DIAMOND-like starting values; the EDM image default is `P_mean = -1.2`. Tune only if the loss-per-σ plots look wrong.
- The network target is `F_target = (y - c_skip·x) / c_out`, with loss `mean((F - F_target)²)`. This equals EDM's weighted loss with `λ(σ) = (σ²+σ_d²)/(σ·σ_d)²`.
- Context frames are **not** scaled by `c_in`. They are concatenated as they are, after the optional augmentation in section 4.8.

`EDMDenoiser(net, cfg)` provides:
- `denoise(x, sigma, context, actions, ctx_level) -> D` (shape (B,C,H,W))
- `loss(target, context, actions, ctx_level) -> scalar` (samples σ and ε internally)
- `sigma` is shape (B,) on the same device as `x`.
- `context` may be (B,K,C,H,W), as `WindowDataset` and `predict_next` give it, or already flattened to (B,K·C,H,W).
- `loss_terms(target, context, actions, ctx_level) -> (per_sample_loss (B,), sigma (B,))`; `loss` is its mean. `loss_by_sigma(losses, sigma)` buckets them for logging. σ and ε come from the global torch RNG.
- `EDMDenoiser` applies neither action dropout nor context noise; the trainer does both.

### 4.7 Denoiser U-Net (`wm/models/unet.py`) `[FROZEN signature]`

```python
UNet(cfg: UNetConfig).forward(
    x: Tensor,           # (B, C, H, W)     noisy target already multiplied by c_in
    c_noise: Tensor,     # (B,)             ln(sigma)/4
    context: Tensor,     # (B, K*C, H, W)   context frames, flattened, oldest first
    actions: LongTensor, # (B, K)           values in 0..6 (6 = NULL_ACTION)
    ctx_level: LongTensor# (B, K)           context noise level per frame, 0..n_levels-1
) -> Tensor              # (B, C, H, W)     F
```

Architecture:
- **Input** `torch.cat([x, context], dim=1)` has `C·(K+1)` channels → conv3×3 → `ch[0]`.
- **Conditioning vector** `emb` (dim `emb_dim = 256`) is the sum of:
  - noise: Fourier features of `c_noise` (16 fixed random frequencies with scale 16, sin and cos) → Linear → SiLU → Linear;
  - actions: `Embedding(NUM_ACTIONS + 1, 64)` + learned positional `(K, 64)` → flatten → Linear(K·64, emb_dim);
  - context level: `Embedding(n_levels, 32)` per frame → flatten → Linear(K·32, emb_dim).
  - Then `emb = SiLU(emb)`.
- **ResBlock(C_in, C_out)**: AdaGN → SiLU → Conv3×3 → AdaGN → SiLU → Dropout(0.0 by default) → Conv3×3 (zero-initialized), plus a 1×1 conv skip if `C_in != C_out`.
  - AdaGN = `GroupNorm(min(32, C), C, affine=False)`, then `h·(1 + scale) + shift`, where `(scale, shift) = Linear(emb_dim, 2C)(emb)`. That Linear is zero-initialized.
- **Encoder**: per level, `n_res` ResBlocks; between levels, a stride-2 Conv3×3 downsample. **Middle**: 2 ResBlocks. **Decoder**: mirror, concatenating the encoder skips; nearest-neighbour upsample ×2 then Conv3×3.
- Optional self-attention at the lowest resolution only (`attn_lowest: bool`, default False).
- **Output**: GroupNorm → SiLU → Conv3×3 to C channels, **zero-initialized** (so at initialization F = 0 and D = c_skip·x).
- Assert `H % 2**(len(ch) - 1) == 0`. For **84**: `ch = [64,128,256]` (84 → 42 → 21). For **96**: `ch = [64,128,256,256]` (96 → 48 → 24 → 12).

Configs: `configs/model/small.yaml` (≈10–15M parameters, `n_res=2`) and
`configs/model/base.yaml` (≈30–40M parameters, wider channels). The model prints its
parameter count at construction.

### 4.8 Conditioning: actions, CFG, and context noise

- **Action dropout** (training): with probability `p_action_drop = 0.1` per sample,
  replace that sample's **whole** action sequence with `NULL_ACTION`.
- **Classifier-free guidance** (sampling):
  `D_cfg = D(x; ∅) + w·(D(x; a) - D(x; ∅))`. Compute both in a single forward pass with the
  batch doubled. `w = 1` means no guidance and must take the plain path (one forward pass).
  GameNGen guided on past frames; we guide on actions because controllability is a main
  evaluation axis.
- **Context noise augmentation** (GameNGen-style anti-drift). Per sample, draw
  `level ~ Uniform{0..n_levels-1}` with `n_levels = 10`, set
  `σ_aug = level / (n_levels - 1) · aug_max` with `aug_max = 0.7` (in [-1,1] pixel units),
  and use `context ← context + σ_aug·ε`.
  - `ctx_noise_mode = "shared"` (default): one level per sample, repeated across the K frames.
  - `"per_frame"`: an independent level per frame, a lightweight diffusion-forcing-style ablation.
  - `aug_max = 0` disables it; level 0 is always passed in that case.
  - At inference `ctx_level` is a parameter (default 0, swept in CP4).

### 4.9 Sampler (`wm/diffusion/sampler.py`)

- **Karras schedule**: `σ_i = (σ_max^(1/ρ) + i/(N-1)·(σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ` for `i = 0..N-1`, then append `σ_N = 0`.
  - Defaults: `σ_min = 2e-3`, `σ_max = 5.0`, `ρ = 7`, `N = 3`. For N = 1 use `[σ_max, 0]`.
- Start from `x_0 = σ_0·ε`.
- **Euler** (default): `d = (x - D(x; σ_i))/σ_i`, then `x ← x + (σ_{i+1} - σ_i)·d`.
- **Heun** (option): Euler step, then a 2nd-order correction on every step except the one that ends at σ = 0.
- Clamp the final output to [-1, 1].
- `sample(denoiser, context, actions, ctx_level, cfg: SamplerConfig, cfg_scale=1.0, generator=None) -> (B,C,H,W)`.

### 4.10 Inference API (`wm/inference.py`) `[FROZEN]`

```python
class WorldModelProtocol(Protocol):
    K: int; C: int; H: int; W: int
    def predict_next(self, context: Tensor,   # (B,K,C,H,W) in [-1,1]
                     actions: Tensor,          # (B,K) long; actions[:, -1] produces the new frame
                     num_steps: int | None = None, cfg_scale: float | None = None,
                     ctx_level: int = 0, generator: torch.Generator | None = None
                     ) -> Tensor: ...          # (B,C,H,W) in [-1,1]

class WorldModel(WorldModelProtocol):
    @classmethod
    def from_checkpoint(cls, path, device=None, use_ema=True, compile=False): ...
```

`WorldModel` runs under `eval()` and `torch.inference_mode()`. `CopyLastFrame`
(`wm/eval/baselines.py`) implements the same protocol, and every eval script accepts either.

### 4.11 Evaluation protocol

- **Metrics** run on [0,1] float tensors with shape (N,3,H,W). Greyscale models are
  repeated to 3 channels for LPIPS.
  - PSNR: `data_range=1`, capped at 100 dB for identical images.
  - SSIM: `torchmetrics`, `data_range=1`.
  - LPIPS: `lpips.LPIPS(net="alex")`, with inputs mapped to [-1,1], at native resolution.
- **Single-step (teacher-forced)**: on val windows, predict the target from real context.
- **Rollout**: for an eval trajectory with frames `f_0..f_N` and actions `a_0..a_{N-1}`,
  the initial context is `f_0..f_{K-1}`. Generated frame `g_j` (for `j ≥ K`) uses the last
  K frames (real at first, generated afterwards) and actions `a_{j-K}..a_{j-1}`.
  Compare `g_j` with `f_j`. The horizon is `h = j - K + 1`.
  - Report at `h ∈ {1,2,5,10,20,50,100,200}`, as mean and 95% CI over ≥64 trajectories.
  - Always use EMA weights, a fixed generator seed, and the default sampler unless the sweep varies it.
- **Baselines**: always include copy-last-frame.
- **Caveat for the writeup**: the emulator has hidden state (RNG, timers) that the frames
  don't reveal, such as when aliens shoot. Pixel metrics will degrade with horizon even for a
  perfect model. LPIPS and the controllability metrics are more meaningful at long horizons.

### 4.12 Configs, seeds, logging, checkpoints

- YAML configs are loaded into dataclasses (`wm/config.py`: `load_config(path, overrides: list[str])`,
  with overrides like `train.lr=2e-4`). Every script takes `--config` and `--set key=value` arguments.
- Every run creates `runs/<YYYYmmdd-HHMM>_<name>/` containing the resolved `config.yaml`,
  `git_commit.txt`, `tb/` (TensorBoard), `checkpoints/`, and `samples/`.
- A checkpoint is a dict: `{"model", "ema", "optimizer", "scheduler", "step", "config", "rng_states"}`.
  Files: `ckpt_last.pt` plus `ckpt_<step:07d>.pt` every `save_every` steps.
- Seed `random`, `numpy`, and `torch` from the config. DataLoader workers get
  `seed + worker_id` via `worker_init_fn`.

### 4.13 Testing conventions

- Use `pytest` with markers registered in `pytest.ini`:
  - `slow`: takes more than 30 s.
  - `gpu`: needs CUDA; skip if unavailable.
  - `network`: downloads from the Hugging Face Hub, PyPI weights, etc.
  - `env`: needs ALE; fast, CPU only.
- The fast suite `pytest -m "not slow and not gpu and not network"` must pass in under 3
  minutes on a laptop CPU.
- Tests use **tiny synthetic data**: `tests/fixtures.py::make_fake_shard(path, n_episodes, lengths, H, W, encode_rows=True)`.
  - With `encode_rows`, each frame stores its global row index in its pixels (e.g. R = row % 256, G = (row // 256) % 256, B = episode_idx % 256). Alignment tests can then decode exactly which row was returned.
- Tests use a tiny net, `tests/fixtures.py::TinyDenoiserNet`, which has the same forward
  signature as the U-Net, so training and sampling tests don't depend on the real U-Net.
- Headless pygame tests set `os.environ["SDL_VIDEODRIVER"] = "dummy"`.

### 4.14 Pitfalls checklist (read before every workflow)

1. **Action alignment.** `actions[i]` is taken at frame `i` and produces frame `i+1`. Off-by-one errors here silently destroy controllability.
2. **Windows must never cross an episode boundary.** The train/val split is by episode.
3. **cv2 conventions.** `cv2.resize` takes `(W, H)`. `cv2.imwrite` expects BGR, so use `imageio` or `matplotlib` to save RGB images.
4. **uint8 overflow.** Convert to int16 or float before subtracting or adding frames, e.g. for diff images.
5. **HDF5 and multiprocessing.** Open files lazily inside each DataLoader worker; never pass an open handle across `fork`.
6. **Gymnasium 1.x** needs `gym.register_envs(ale_py)` before `gym.make("ALE/...")`.
7. **SB3 Zoo agents** were trained on `SpaceInvadersNoFrameskip-v4` with 84×84 greyscale, 4-frame stacks. Feed them `AgentObsStack`, never the model frames.
8. **Sampling hygiene.** Sample with EMA weights, `model.eval()`, and under `inference_mode`, then clamp outputs. CFG with `w=1` must not double the batch.
9. **pygame `surfarray`** expects (W,H,3), so call `frame.swapaxes(0,1)`.
10. **Determinism.** Don't introduce randomness in `make_env` that isn't derived from the seed.
11. **Do not aspect-correct, crop, or otherwise alter stored frames** beyond `to_model_frame`.

### 4.15 Training on Discovery (Slurm)

- Training, evaluation rollouts, and benchmarks run on Dartmouth's Discovery cluster
  through Slurm batch jobs. Data collection and the play scripts can run on laptops.
  Partition names, GPU types, and time limits come from the Discovery documentation;
  record them in section 1.4 and keep them out of the code (pass them as `sbatch` options).
- **Environment on the cluster:** load a Python 3.13 module (`setup.sh` requires
  `python3.13`), create the `.venv` with `setup.sh`, then install the CUDA build of PyTorch. Never import pygame from
  training or evaluation code, so jobs run headless.
- **Job template** `scripts/slurm/train.sbatch`: requests one GPU, activates the venv,
  and runs `exec python scripts/train.py --config $CONFIG --resume auto`, writing logs to
  `runs/slurm-%j.out`. It sets `#SBATCH --signal=B:USR1@300` so the trainer saves a
  checkpoint 5 minutes before the time limit. The `exec` matters: it makes Python the
  batch shell itself, so the signal reaches the trainer.
- **Auto-resume:** `--resume auto` picks up `ckpt_last.pt` from the most recent run with
  the same name if one exists. The trainer also saves on SIGUSR1 and SIGTERM. Long
  runs are a chain of jobs (`sbatch --dependency=afterany:<jobid>`), and none of them
  lose more than `save_every` steps.
- **Data on the cluster:** copy `data/` to fast scratch storage before training, and point
  configs at it with `--set data.root=...`. Never write HDF5 shards to the same file from
  more than one job.
- **Access:** only Syed has a Discovery account. Any GPU work from another track (the
  CP3-C GPU timing test, CP4-C sampler benchmarks, CP6-C ablation runs) is handed over as
  a config file plus the exact `sbatch` command in that workflow's progress note. Syed
  submits it and shares the outputs (checkpoint paths, logs, result JSON). Code for those
  workflows must also run on CPU, so it can be developed and tested without the cluster.
- Keep checkpoints small to move: prune all but the last few `ckpt_<step>.pt` files
  (`keep_last: 3` in the train config), and always keep the EMA weights.

---

## 5. Phases

Timeline, aligned with the proposal's schedule (term weeks):

| Weeks | Checkpoints | Proposal milestone |
|---|---|---|
| 6–7 | P0, CP1, CP2 | Record the data, check the laser survives, score the baseline |
| 8 | CP3 | First small model on Discovery that beats the baseline |
| 9–10 | CP4 | Autoregressive rollouts and context noise augmentation |
| 11–12 | CP5 | Action conditioning with CFG, human play recorded |
| 13 | CP6 (demo, ablations) | Playable demo, ablations on context length and sampling steps |
| 14 | CP6 (final evaluation) | Final results, write-up, presentation |
| if time | CP7 | Stretch |

### Phase 0: Repo skeleton and frozen interfaces (one person, sequential, ~1 day)

**Branch:** `cp0/skeleton`. **Owns:** everything it creates.

Tasks:
1. Create the layout from section 4.1. Move and write the play scripts:
   - Move `play.py` → `scripts/play_gym.py` (fix or drop its no-op `WideRender`, see section 1.2).
   - Write `scripts/play_real.py` from scratch on `wm.env` and `wm.controls`: a custom pygame
     loop showing the max-pooled frame (no flicker), `FireLatch` so quick Space taps register,
     an `--fps` option, and ×2 horizontal widening for display only.
   - `scripts/inspect_frames.py` is not created here; CP1-A writes it.
2. Write `wm/data/schema.py` in full (action constants, source codes, end-reason codes, dataset names and dtypes, `SCHEMA_VERSION = 1`).
3. Write `wm/controls.py` in full: `keys_to_action(left: bool, right: bool, fire: bool) -> int`
   (left and right together cancel out) and `ACTION_TO_MOVE_FIRE = {0:(0,0), 1:(0,1), 2:(1,0), 3:(2,0), 4:(1,1), 5:(2,1)}`,
   where move is `0=none, 1=right, 2=left`. Also a `FireLatch(n=4)` helper.
4. Write `wm/env.py` in a working first version: `MaxPoolSkip`, `make_env` (seeded no-ops), `to_model_frame("area")`, `to_agent_obs`, `AgentObsStack`. CP1-A hardens and tests it.
5. Create stub modules for every other file in `wm/`, with the `[FROZEN]` signatures from section 4 raising `NotImplementedError`.
6. Write `wm/config.py`, `pytest.ini` (markers), `tests/conftest.py`, and `tests/fixtures.py` (`make_fake_shard` may stay a stub raising `NotImplementedError` until CP1-B; `TinyDenoiserNet` implemented in full).
7. Update `requirements.txt` (section 4.2), `.gitignore`, and the README (setup including the PyTorch install, how to play). Create `progress/` with `TEMPLATE.md`.

Tests:
- `tests/test_smoke.py` imports every module.
- `keys_to_action` covers all 8 key combinations.
- `make_env(0)` (marker `env`): reset obs is (210,160,3) uint8, and one step works.

**Done when** the fast suite passes and both `python scripts/play_gym.py` and `python scripts/play_real.py` are playable. Tag `cp0`.

---

### CP1: Data foundation

#### CP1-A: Environment module and resolution decision
**Branch:** `cp1/A-env` · **Owns:** `wm/env.py`, `scripts/inspect_frames.py`, `tests/test_env.py`, `docs/figures/cp1_*`

Tasks:
1. Harden `MaxPoolSkip` and `make_env` to match section 4.3 exactly (early termination, `info` fields, episode seed counter, truncation).
2. Add the `"max"` method to `to_model_frame`.
3. Write `scripts/inspect_frames.py` (new; no earlier version exists) on top of `wm.env`. It should produce `docs/figures/cp1_frames.png`, a grid of raw frames and {84, 96} × {area, max} × {RGB, grey} versions with 4× nearest upscaling, plus zoomed crops around the player's laser.
4. Write a **laser visibility check** (`scripts/inspect_frames.py --laser-stats`):
   - Run a seeded firing rollout of 300 steps.
   - Find the player-laser pixels in the raw max-pooled frame by their colour in the play area between the shields and the aliens (calibrate the colour once by frame differencing right after FIRE, and record it in the script).
   - For each processed version, count a frame as "visible" if the downscaled laser region has a pixel whose maximum channel difference from the background is ≥ 40.
   - Report the visibility rate for each version, and also for the raw single last frame without max-pooling, to quantify the flicker.
5. **Decision rule for D10:** choose the smallest resolution and method with ≥ 95% laser visibility, preferring `area` at 84 if it passes. Write the result in the progress note.

Tests (`env` marker):
- Shapes and dtypes of every transform.
- **Determinism:** two envs, same seed and the same 200 random actions ⇒ identical observations, rewards, and lives.
- Reward from `MaxPoolSkip` equals the sum over 4 manual base steps.
- Early termination returns a valid obs.
- `info["noops"]` is reproducible from the seed and stays in `[0, 30]`.
- In a seeded firing rollout, the max-pooled frames show the laser at least as often as single raw frames.
- `AgentObsStack` shifts correctly and zero-fills on reset.

**Done when:** the tests pass, the figure exists, and the D10 decision is written in the progress note with its numbers.

#### CP1-B: HDF5 storage
**Branch:** `cp1/B-storage` · **Owns:** `wm/data/writer.py`, `wm/data/reader.py`, `scripts/validate_data.py`, `tests/test_storage.py`, `tests/fixtures.py::make_fake_shard`

Tasks:
1. `ShardWriter` and `ShardReader` exactly as in section 4.4: resizable datasets, buffered writes (flush every 1024 rows), rotation between episodes, attrs including `git_commit` (use `git rev-parse HEAD`, falling back to "unknown").
2. `scripts/validate_data.py <dataset_dir>`: checks all invariants and prints statistics (rows, episodes, by source, mean episode length, action histogram, disk size, compression ratio). Exits non-zero on any violation.
3. Implement `make_fake_shard` in full, using the writer itself.

Tests:
- Round trip: write 3 episodes with lengths [5, 1, 40]; every field reads back exactly. A 1-row episode, with 0 steps, is legal.
- Rotation creates a new file only between episodes.
- The validator catches injected errors: a wrong -1 position, a gap between episodes, a misplaced terminal.
- `ShardReader` works after pickling (lazy open), as DataLoader workers require.
- Write speed of ≥ 2,000 rows/s at 84×84 (marker `slow`).

**Done when:** the tests pass and `validate_data.py` runs on a fake shard.

#### CP1-C: Policies
**Branch:** `cp1/C-policies` · **Owns:** `wm/agents/*`, `scripts/eval_agent.py`, `tests/test_policies.py`

Interface (`wm/agents/policies.py`):

```python
class Policy(Protocol):
    def reset(self, seed: int, raw_obs: np.ndarray) -> None: ...
    def act(self, raw_obs: np.ndarray) -> int: ...       # raw_obs = MaxPoolSkip output (210,160,3)
```

Tasks:
1. `RandomPolicy(mean_hold=4)`: samples a uniform action and holds it for `1 + Geometric` steps with the given mean, which gives more natural movement than i.i.d. actions.
2. `EpsilonMixPolicy(base, epsilon, mean_hold=2)`: with probability ε per step, starts a random action held for a short run; otherwise defers to `base`. The base policy still observes every frame so its frame stack stays correct.
3. `ZooAgentPolicy` in `wm/agents/pretrained.py`:
   - Load with `huggingface_sb3.load_from_hub(repo_id, filename)` and `PPO.load(path, custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0}, device="cpu")`.
   - Default `repo_id = "sb3/ppo-SpaceInvadersNoFrameskip-v4"`. Verify the repo exists. Fallbacks are the `sb3/dqn-` or `sb3/qrdqn-` versions (QR-DQN needs `sb3-contrib`; append it to requirements if used).
   - Keep an internal `AgentObsStack` fed by `to_agent_obs(raw_obs)`. Transpose to match `model.observation_space.shape`, which is (84,84,4) or (4,84,84); assert which one it is.
   - Use `predict(..., deterministic=False)` for diversity.
4. `scripts/eval_agent.py --episodes 5`: mean and std of score for the agent, ε=0.2, ε=0.3, and random. Use `make_env` from Phase 0.

Tests:
- `RandomPolicy` hold lengths average 4 ± 0.5 over 10k steps, and actions are uniform within tolerance.
- `EpsilonMixPolicy` with ε=0 equals the base policy and with ε=1 is never the base.
- Agent loading and `act` return an int in 0..5 (markers `network`, `env`).
- Agent mean score ≥ 2× random over 3 episodes (markers `slow`, `network`, `env`).

**Done when:** the tests pass and the score table is in the progress note.

**CP1 integration checks (integrator):**
1. `scripts/smoke_cp1.py`: the agent at ε=0.2 plays 2 episodes, capped at 500 steps, written with `ShardWriter` at the D10 resolution. `validate_data.py` passes on the output.
2. Visually inspect 20 frames from that shard, upscaled: the laser is visible and the frames are not aspect-corrected.
3. Record D10 and the agent scores in section 1.

---

### CP2: Datasets and diffusion core

#### CP2-A: Collector, debug and eval datasets, coverage report
**Branch:** `cp2/A-collect` · **Owns:** `scripts/collect.py`, `configs/data/*.yaml`, `wm/data/ram.py`, `wm/data/coverage.py`, `scripts/coverage_report.py`, `tests/test_collect.py`, `tests/test_coverage.py`

Tasks:
1. `scripts/collect.py --config configs/data/debug.yaml --workers N`:
   - The config lists jobs, each with policy, epsilon range, fraction of steps, and seed range. The total step target is split across workers.
   - Each episode's seed is `base_seed + worker * 10_000_000 + local_counter`. ε is drawn per episode from the job's range and stored.
   - Uses `multiprocessing` with spawn; each worker owns its own writer.
   - Resume-safe: skip finished shards; write `_DONE` markers.
   - Writes a `summary.json`.
2. `configs/data/debug.yaml`: 500k steps; 50% agent ε=0.2, 25% agent ε=0.3, 25% `RandomPolicy`.
3. `configs/data/eval.yaml`: ~100 episodes, each truncated at 400 steps, with seeds from 1,000,000,000; 70% agent ε=0.2 and 30% random; written to `data/eval/`.
4. `wm/data/ram.py`: a verified RAM address map for Space Invaders.
   - Start from the AtariARI annotations (Anand et al., 2019), which include invaders-left count, player x, lives, and score.
   - **Verify each address** by plotting it against frames before use (`scripts/coverage_report.py --verify-ram` saves `docs/figures/cp2_ram_verify.png`).
   - Drop any address that doesn't verify.
5. `wm/data/coverage.py` and `scripts/coverage_report.py <dataset_dir>` write a markdown report and montages to `docs/figures/cp2_coverage_*`:
   - Histograms of score, episode length, invaders left, lives, and source.
   - Counts of life losses, game overs, and wave transitions (invaders-left jumps back up).
   - UFO presence: a pixel-colour heuristic in the top band, or RAM if verified.
   - Shield erosion: shield-colour pixel count relative to episode start.
   - Montages of 16 random frames per bucket so a person can check the heuristics.
6. **Initial coverage targets** (revise them if they prove unrealistic, and say why):
   - ≥ 5% of frames with ≤ 6 invaders left
   - ≥ 50 wave transitions (revised in CP2 to ≥ 20 for the debug set; ≥ 50 stays for `data/full/`, see D17)
   - ≥ 200 game overs
   - UFO visible in ≥ 1% of frames
   - ≥ 10% of frames with shields ≤ 50% intact
7. Collect `data/debug/` and `data/eval/`.

Tests:
- Tiny collect (2 workers × 300 steps; `env` marker): the validator passes, sources and ε are recorded, seeds are unique.
- **Replay determinism (critical):** for a collected episode, `make_env(seed)`, then replaying the stored actions, reproduces every stored frame exactly (after `to_model_frame`).
- The coverage report runs on a fake shard.

**Done when:** the debug and eval sets exist and validate, the coverage report is committed under `docs/`, and any gaps against the targets are written up in the progress note.

#### CP2-B: PyTorch window dataset and loader
**Branch:** `cp2/B-dataset` · **Owns:** `wm/data/dataset.py`, `scripts/bench_loader.py`, `tests/test_dataset.py`

Tasks:
1. `WindowDataset` exactly as in section 4.5. Precompute the valid target rows per shard (int64), with a global index built from cumulative offsets.
2. `make_dataloader(cfg, split)` with `worker_init_fn` (seeding plus lazy HDF5 open), `pin_memory`, `persistent_workers`, and `drop_last` for training.
3. Optional `in_memory=True`: load all frames into RAM when they fit, for the debug set.
4. `scripts/bench_loader.py`: samples per second for `num_workers ∈ {0, 2, 4, 8}`. Target ≥ 2,000 samples/s at K=4, 84×84.

Tests (on `make_fake_shard` with `encode_rows`):
- **Exact alignment:** for random items, the decoded context rows are `i-K..i-1`, the actions match `actions[i-K..i-1]`, and the target row is `i`.
- No window crosses an episode boundary; the window count is Σ max(0, length − K).
- The train and val splits share no episode and are deterministic.
- Output range is [-1,1], dtypes and shapes are correct, and greyscale gives C=1.
- Works with `num_workers=2`.
- K=1 and K=8 edge cases.

**Done when:** the tests pass and benchmark numbers are in the progress note.

#### CP2-C: EDM core and sampler
**Branch:** `cp2/C-edm` · **Owns:** `wm/diffusion/*`, `configs/train/edm_default.yaml`, `tests/test_edm.py`, `tests/test_sampler.py`

Tasks:
1. `EDMConfig` and `EDMDenoiser` exactly as in section 4.6 (preconditioning, σ sampling, loss, `denoise`).
2. `SamplerConfig`, `karras_sigmas`, and `sample` (Euler and Heun, CFG hook) as in section 4.9. Context noise is handled outside the sampler, by passing `ctx_level`.
3. A loss-by-σ helper that returns the loss per log-σ bucket, for trainer logging.

Tests (using `TinyDenoiserNet` or analytic oracles, no U-Net):
- Coefficient identities: `c_skip(σ→0) → 1`, `c_out(σ→0) → 0`.
- With the oracle `F = F_target`, the loss is 0.
- **Analytic oracle sampling:** for data `y ~ N(0, s²)` (per pixel, with s ≠ σ_d), the optimal denoiser is `D(x;σ) = x·s²/(σ²+s²)`. Plug it into `sample` with N=50 Heun steps: the samples have std within 3% of s. With N=3 Euler, check that the result is finite and its std is within 3% of the closed-form Euler std (≈ 0.53·s on the default schedule; Euler at N=3 cannot reach s, see `progress/cp2-C.md`).
- The schedule is monotone decreasing, ends at 0, and the first value equals σ_max.
- `cfg_scale=1` makes exactly one network call per step (count the calls with a mock).
- `cfg_scale=0` equals the unconditional result.
- Determinism with a seeded generator.

**Done when:** the tests pass.

**CP2 integration checks:**
1. The DataLoader over `data/debug/` plus `EDMDenoiser(TinyDenoiserNet)` trains for 200 steps on CPU with no NaN, and the loss decreases.
2. Copy the coverage summary and dataset size into section 1.5.

---

### CP3: Single-step model (first real training)

#### CP3-A: Metrics, baselines, visualization, human recorder
**Branch:** `cp3/A-eval` · **Owns:** `wm/eval/metrics.py`, `wm/eval/baselines.py`, `wm/eval/visualize.py`, `scripts/eval_single_step.py`, `scripts/play_record.py`, `tests/test_metrics.py`, `tests/test_play_record.py`

Tasks:
1. Metrics as in section 4.11 (batched, on any device; the LPIPS model loads lazily and is cached).
2. `CopyLastFrame` implementing `WorldModelProtocol`.
3. `visualize.save_grid(path, context, target, pred)`: columns for the last context frame, target, prediction, and |diff|×4; nearest upscaling ×4; saved as PNG.
4. `scripts/eval_single_step.py --model <ckpt|copy_last> --n 2048`: teacher-forced metrics on the val split, saving JSON and a grid.
5. `scripts/play_record.py` (human data):
   - Uses `make_env` and `ShardWriter` with `source=human`, at the D10 resolution, into `data/human/`.
   - One agent step per tick at `--hz 15` (real speed) or lower.
   - Uses `wm.controls` (with `FireLatch`); displays raw max-pooled frames with ×2 horizontal widening for display only.
   - Esc ends the episode with `end_reason=stopped`.
   - Seeds are drawn from a separate range starting at 2,000,000,000.

Tests:
- PSNR of identical images = 100 (cap); a constant offset of 0.1 gives 20 dB.
- SSIM of identical images = 1.
- LPIPS of identical images ≈ 0 (`network`).
- On a fake shard, `CopyLastFrame` gives exactly the PSNR between consecutive encoded frames.
- `play_record` headless with a scripted key sequence writes a valid shard (`env`; SDL dummy driver).

**Done when:** the tests pass and the copy-last baseline numbers on debug val are in the progress note.

#### CP3-B: Trainer
**Branch:** `cp3/B-trainer` · **Owns:** `wm/train/trainer.py`, `scripts/train.py`, `scripts/slurm/*`, `configs/train/debug.yaml`, `tests/test_trainer.py`

Tasks:
1. `scripts/train.py --config configs/train/debug.yaml [--set ...] [--resume RUN_DIR] [--dry-run]`.
2. Training setup:
   - Optimizer: AdamW (lr 1e-4, betas (0.9, 0.999), weight decay 1e-2), linear warmup over 1,000 steps, then constant (cosine optional).
   - Gradient clipping at 1.0. Optional gradient accumulation.
   - bf16 autocast when CUDA supports it, otherwise fp32.
   - EMA with decay 0.999, updated every step after warmup.
   - Optional `torch.compile` flag.
3. Action dropout as in section 4.8 (`p_action_drop`, default 0.1). Add the config keys `aug_max` (default 0) and `ctx_noise_mode`, but leave the augmentation itself to CP4-B. In CP3 the trainer always passes `ctx_level = zeros(B, K)`.
4. Logging to TensorBoard: loss, loss per σ bucket, learning rate, gradient norm, throughput. Validation loss every `val_every` steps. Every `sample_every` steps, save a sample grid of 8 val windows using EMA weights and the default sampler.
5. Checkpointing and resume as in section 4.12. Exact resume restores RNG states.
   Add `--resume auto` and save-on-signal (SIGUSR1, SIGTERM) as in section 4.15.
6. `scripts/slurm/train.sbatch` and `scripts/slurm/eval.sbatch` as in section 4.15.
7. `configs/train/debug.yaml`: K=4, batch 64, 50k steps, model `small`, data `data/debug/`.

Tests (with `TinyDenoiserNet` and a fake shard, on CPU):
- Overfitting one fixed batch for 300 steps reduces the loss by ≥ 5×.
- Resume: 10 steps, save, resume to 20 steps equals 20 straight steps (fp32, deterministic, `allclose`).
- The empirical action drop rate is 0.1 ± 0.03.
- The checkpoint has all keys.
- `--dry-run` runs 5 steps.
- Sending SIGUSR1 to a training process writes `ckpt_last.pt` and exits cleanly, and `--resume auto` continues from it.

**Done when:** the tests pass and a 1,000-step debug job has run on Discovery through `train.sbatch`.

#### CP3-C: U-Net denoiser
**Branch:** `cp3/C-unet` · **Owns:** `wm/models/embeddings.py`, `wm/models/unet.py`, `configs/model/small.yaml`, `configs/model/base.yaml`, `tests/test_unet.py`

Tasks:
1. Implement section 4.7 exactly, including `UNetConfig(C, K, H, ch, n_res, emb_dim, n_levels, attn_lowest, dropout)`.
2. Two configs, `small` and `base`, for both 84 and 96. Report parameter counts in the progress note.

Tests:
- Output shapes for H ∈ {84, 96}, K ∈ {1, 4, 8}, C ∈ {1, 3}.
- At initialization the output is exactly zero.
- After re-initializing the output conv with non-zero weights, changing `actions`, `c_noise`, `ctx_level`, or `context` each changes the output.
- After one backward pass every parameter has a gradient (no unused parameters).
- An indivisible H raises a clear error.
- Forward time at batch 64 (marker `gpu`).

**Done when:** the tests pass.

**CP3 integration checks (first real run):**
1. Train `configs/train/debug.yaml` with the real U-Net for 50k steps on Discovery (chaining jobs if needed). Attach the TensorBoard curves to the progress notes.
2. `eval_single_step.py` on val: **the model's PSNR and LPIPS must beat copy-last-frame.** Record both in section 1.5.
3. In the sample grids, the player, aliens, shields, score, and lasers are all recognisable. If the lasers are missing, revisit D10 before CP4.

---

### CP4: Autoregressive rollouts and anti-drift

#### CP4-A: Rollouts and horizon evaluation
**Branch:** `cp4/A-rollout` · **Owns:** `wm/eval/rollout.py`, `wm/eval/plots.py`, `scripts/eval_rollouts.py`, `tests/test_rollout.py`

Tasks:
1. `rollout(model: WorldModelProtocol, frames0 (B,K,C,H,W), actions (B, K+T-1), T, **sample_kwargs) -> (B,T,C,H,W)`, using exactly the alignment in section 4.11.
2. `scripts/eval_rollouts.py --models <ckpt...|copy_last> --T 200 --n 64`:
   - Loads `data/eval/` trajectories.
   - Writes per-horizon metrics as CSV and JSON.
   - Plots PSNR, SSIM, and LPIPS against horizon (log x axis), with 95% CIs, for all models on one figure.
   - Exports 4 side-by-side videos (real | generated | |diff|) as MP4 at 15 fps with ×4 nearest upscaling (display widening optional).

Tests:
- With `CopyLastFrame`, every generated frame equals the last context frame.
- **Index-alignment fixture:** a fake model that outputs "last frame's encoded index + 1" generates frames whose decoded index equals j.
- A fake model that writes `actions[:, -1]` into the pixels shows that frame j encodes `a_{j-1}`.
- Metrics against ground truth are perfect when a cheating oracle model returns `f_j`.

**Done when:** the tests pass and the CP3 model's horizon curve with the copy-last baseline is in the progress note.

#### CP4-B: Context noise augmentation runs
**Branch:** `cp4/B-noiseaug` · **Owns:** noise-augmentation code in `wm/train/trainer.py` (B owned the trainer in CP3), `configs/train/debug_aug.yaml`, `configs/train/debug_noaug.yaml`, `tests/test_noise_aug.py`

Tasks:
1. Implement context noise augmentation as in section 4.8 (`shared` and `per_frame` modes). Log the distribution of levels.
2. Train two debug models that are identical except `aug_max` ∈ {0, 0.7}, with the same seed, data, and steps.
3. Hand off both checkpoints (paths in the progress note). The horizon comparison runs at integration, using CP4-A's script.

Tests:
- `aug_max=0` leaves the context unchanged.
- The empirical std of `(ctx_aug - ctx)` matches `σ_aug` per level (±5%).
- In `shared` mode, all frames in a sample get the same level; `per_frame` levels vary.
- Level 0 means no noise.

**Done when:** the tests pass and both runs have finished training.

#### CP4-C: Inference engine and sampler benchmarks
**Branch:** `cp4/C-inference` · **Owns:** `wm/inference.py`, `scripts/bench_sampling.py`, `tests/test_inference.py`

Tasks:
1. `WorldModel` as in section 4.10: loads a checkpoint plus config, uses EMA weights, `channels_last`, optional bf16/fp16 autocast, optional `torch.compile`, and batched CFG.
2. `scripts/bench_sampling.py`:
   - Measures steps/s at batch 1 for `num_steps ∈ {1, 2, 3, 5, 10}` × {Euler, Heun}, on GPU and CPU.
   - Measures single-step PSNR and LPIPS on 512 val windows at each setting, computing the metrics inline with `wm/eval/metrics.py`.
   - Outputs a table and a plot of quality vs speed.
3. Propose the D11 default number of steps: the fewest steps within 0.3 dB PSNR of 10 steps that runs at ≥ 15 steps/s on the training GPU.

Tests (fake checkpoint with `TinyDenoiserNet`):
- Shapes are correct and output is in [-1,1].
- Results are deterministic with a generator.
- `cfg_scale=1` matches the no-CFG path exactly.
- Batched prediction matches per-item prediction (`allclose`).
- EMA weights load when `use_ema=True`.

**Done when:** the tests pass and the benchmark table is in the progress note.

**CP4 integration checks:**
1. Run `eval_rollouts.py` on copy-last, the aug-off model, and the aug-on model, with the inference context level swept over {0, 1, 2}. Save the figure as `docs/figures/cp4_horizon.png`. **This is the first version of the project's main figure.**
2. Expected result: aug-on degrades more slowly beyond ~20 steps. If not, record it and check the alignment tests and the `ctx_level` plumbing before going further.
3. Record the D11 steps default and the best inference context level.

---

### CP5: Action control and full-scale training

Merge order at the end of CP5 is **C, then A, then B**, because A's final controllability
table uses C's CFG and B's final checkpoint.

#### CP5-A: Controllability evaluation and inverse dynamics
**Branch:** `cp5/A-control` · **Owns:** `wm/eval/detectors.py`, `wm/eval/counterfactual.py`, `wm/models/idm.py`, `scripts/eval_counterfactual.py`, `scripts/train_idm.py`, `tests/test_control.py`

Tasks:
1. Pixel detectors for **generated** frames:
   - `player_x(frame)`: centroid of player-coloured pixels in the bottom band.
   - `laser_present(frame)`.
   - Calibrate `player_x` against RAM player x on real data and report the MAE in pixels.
2. Counterfactuals (`scripts/eval_counterfactual.py`):
   - Take 64 eval trajectories and pick a branch point t ≥ K in each.
   - Action programs: constant NOOP, LEFT, RIGHT, FIRE, LEFTFIRE, RIGHTFIRE for 30 steps.
   - **Real reference:** replay seed + no-ops + actions up to t in the real emulator, then apply the program. Determinism makes this exact.
   - **Model:** roll out from the real context with the same program.
   - Metrics:
     - Ordering accuracy `x_LEFT < x_NOOP < x_RIGHT` at step 30, with a 2-pixel margin.
     - Mean Δx per step, model vs real.
     - Fire response: P(laser appears within 3 steps | FIRE pressed and no laser on screen), model vs real.
3. Inverse dynamics model (`wm/models/idm.py`):
   - A small CNN on `concat(f_t, f_{t+1})`, with two heads: move (3 classes) and fire (2 classes), using `ACTION_TO_MOVE_FIRE`.
   - Train on the train split; report accuracy on real val. This is the ceiling: fire is often unobservable when a laser is already in flight.
   - Then report accuracy on model rollouts with known actions.

Tests:
- Detectors on synthetic frames with known sprite positions.
- The counterfactual script runs end to end with `CopyLastFrame` (ordering accuracy ≈ 0, as expected).
- The IDM overfits 256 fake pairs.
- Mapping table coverage.

**Done when:** the tests pass and results for the CP4 aug-on model are in the progress note.

#### CP5-B: Full dataset and final training run
**Branch:** `cp5/B-fullrun` · **Owns:** `configs/data/full.yaml`, `configs/train/final.yaml`, `scripts/slurm/final_chain.sh` (submits the chained final-run jobs)

Tasks:
1. Start this on **day 1** of CP5, since collection takes a long time.
   - `configs/data/full.yaml`: 2–5M steps (choose based on disk; at 84×84 with LZF, expect roughly 4–8 GB per million rows, so measure on the debug set).
   - Same mixture as debug, adjusted using the CP2 coverage gaps (e.g. more low-ε agent episodes for late waves).
   - Include all `data/human/` shards.
2. Validate all shards and run the coverage report against the targets.
3. `configs/train/final.yaml`:
   - Model `base`, K=4, `aug_max=0.7` (shared mode), action dropout 0.1, batch 64–128.
   - Steps: as many as the compute budget allows, saving checkpoints every 10k.
   - Launch it. Action dropout is already in the trainer, so this does not need to wait for CP5-C.
4. Monitor: sample grids, val loss, and a quick `eval_rollouts.py --n 16` on intermediate checkpoints.

Tests:
- `train.py --config configs/train/final.yaml --dry-run` passes.
- `validate_data.py` passes on `data/full/`.

**Done when:** the final run has reached its planned step count, or is far enough along for CP6 to use a checkpoint. Record the checkpoint path in the progress note.

#### CP5-C: Action conditioning and classifier-free guidance
**Branch:** `cp5/C-cfg` · **Owns:** CFG code in `wm/diffusion/sampler.py` and `wm/inference.py`, the optional `action_input` flag in `wm/models/unet.py`, `scripts/action_sensitivity.py`, `tests/test_cfg.py`

Tasks:
1. Confirm the CFG path end to end (sampler, `WorldModel`, rollout kwargs).
2. `scripts/action_sensitivity.py`: a quick metric that doesn't depend on CP5-A. For val contexts, compute the mean absolute pixel difference in the bottom band between predictions under LEFT and RIGHT with a shared noise seed. Sweep `w ∈ {1, 1.5, 2, 3}`.
3. **If the model ignores actions** (sensitivity ≈ copy-last level), add `UNetConfig.action_input: "adagn" | "adagn+planes"`.
   - `"adagn+planes"` also concatenates the one-hot of the latest action as 7 constant input planes.
   - Retrain a debug model with it and compare.
4. Propose the D11 CFG default: the smallest `w` that reaches most of the sensitivity gain without visible artifacts (check grids).

Tests:
- `w=1` gives the same result as no CFG.
- `w=0` gives the unconditional (all-`NULL_ACTION`) result.
- `NULL_ACTION` never appears in dataset outputs.
- The `adagn+planes` variant passes the U-Net shape and gradient tests.

**Done when:** the tests pass and the sweep table is in the progress note.

**CP5 integration checks:**
1. Run `eval_counterfactual.py` and the IDM evaluation on the latest final-run checkpoint with the chosen `w`. Save `docs/figures/cp5_counterfactual.png`, a strip of LEFT/NOOP/RIGHT branches next to the real emulator.
2. Record D11 (CFG) and the controllability table in section 1.5.

---

### CP6: Demo, ablations, final evaluation

#### CP6-A: Final evaluation and figures
**Branch:** `cp6/A-final-eval` · **Owns:** `docs/results.md`, `docs/figures/final_*`, `configs/eval/final.yaml`

Tasks:
1. On the final checkpoint, run single-step metrics, horizon curves (final vs debug aug-on vs copy-last), counterfactuals, IDM, and the speed table.
2. A failure-case gallery: drift examples, hallucinated or vanished aliens, laser errors.
3. `docs/results.md` with every number and figure needed for the writeup.
4. Optional: a small human study where classmates guess real vs generated for 2-second clips.

Checks: every number in `results.md` can be regenerated with a command listed next to it.

#### CP6-B: Playable demo
**Branch:** `cp6/B-demo` · **Owns:** `scripts/play_model.py`, `scripts/make_scenarios.py`, `demo/scenarios.json`, `tests/test_demo.py`

Tasks:
1. `scripts/make_scenarios.py`: selects starting contexts (K real frames plus actions) from `data/eval/` or `data/full/` using RAM:
   - a fresh game
   - ≤ 6 invaders left
   - a UFO about to appear or visible
   - one life left
   - Saves them to `demo/scenarios.json` as `{name, dataset, shard, row}` entries.
2. `scripts/play_model.py --ckpt ...`:
   - pygame window; `wm.controls` with `FireLatch`.
   - Keys: A/D or arrows to move, Space to fire, R to reset the scenario, 1–9 to pick a scenario, +/− to change sampling steps, Esc to quit.
   - Fixed target tick rate (`--hz`, default 15, and fine to run lower).
   - Overlays fps and "every frame generated by a neural network".
   - Display ×4 upscaling with optional ×2 horizontal widening.
   - Optional `--side-by-side` mode that runs the real emulator in parallel from replayable scenarios.

Tests:
- Headless run for 20 ticks with `CopyLastFrame` (SDL dummy driver).
- The scenario file schema is valid and each scenario loads K frames.

**Done when:** a teammate who has never used it can play from the README instructions.

#### CP6-C: Ablations
**Branch:** `cp6/C-ablations` · **Owns:** `scripts/run_ablations.py`, `configs/train/ablate_*.yaml`, `docs/figures/ablation_*`

Run all training ablations at **debug scale** (small model, debug data, same steps and seed).
Start them during CP5 if a GPU is free.

1. Context length K ∈ {1, 4, 8}.
2. Context noise augmentation on/off (reuse the CP4-B runs) and `shared` vs `per_frame`.
3. Sampling steps {1, 2, 3, 5, 10}: inference only, on the final model.
4. CFG `w` (reuse the CP5-C sweep).

For each ablation, run the horizon evaluation (n=64, T=100) and action sensitivity, and
output a table in `docs/results.md` plus plots.

**CP6 integration checks:** the demo runs from a clean clone after `setup.sh` and the
PyTorch install; `docs/results.md` is complete; tag `cp6` as the submission version.

---

### CP7 (stretch): Agent trained inside the world model

Only start this if CP6 is done early.

1. Reward and termination predictors: small CNNs on `(f_t, f_{t+1})` trained on real data.
   - Reward is predicted as a class over the observed reward values.
   - Termination comes from the `terminal` flag.
2. `DreamEnv(gym.Env)`: wraps `WorldModel` plus the predictors. It starts from a dataset scenario, presents 84×84 greyscale 4-frame stacks to the agent, and caps episodes at 1,000 steps.
3. Train PPO (SB3, `CnnPolicy`) in `DreamEnv`, then evaluate in the real env with `make_env`. Compare against random play and the Zoo agent.

---

## 6. Prompt templates

**Workflow prompt (for Claude Code):**

```
You are implementing workflow CP<N>-<X> ("<name>") of the Space Invaders world model project.
1. Read ROADMAP.md sections 0, 2, 3, 4 in full, then section 5 → CP<N>-<X>.
2. Check out branch cp<N>/<X>-<name> from tag cp<N-1>.
3. Only create/modify the files listed under "Owns". Do not change [FROZEN] interfaces.
4. Write the listed tests first, then implement. Run: pytest -m "not slow and not gpu and not network"
5. Run the workflow's slow/network/gpu tests if the machine supports them and report results.
6. Fill in progress/cp<N>-<X>.md using the template. Commit with prefix [cp<N>-<X>].
Context from the team: <anything extra, e.g. D10 = 84/area, GPU = ...>
Stop and ask if a requirement conflicts with ROADMAP.md section 4.
```

**Integration prompt (for Claude Code, run by the integrator):**

```
You are the integrator for checkpoint CP<N>. Read ROADMAP.md sections 0, 3.3, and 5 → CP<N>.
Merge branches in the stated order into main, running the fast test suite after each merge.
Resolve conflicts following section 3.4. Run the CP<N> integration checks and report results.
Read progress/cp<N>-*.md, then update ROADMAP.md section 1 (status board, decisions,
open questions, results log) and apply any approved interface changes to section 4.
Tag cp<N>.
```

**Fix prompt:**

```
On branch <branch>, test <test name> fails with: <error>. Read ROADMAP.md section 4.14 and the
relevant spec section, find the root cause, fix it without changing [FROZEN] interfaces, and
re-run the fast suite.
```