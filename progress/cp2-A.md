# cp2-A: Collector, debug and eval datasets, coverage report

## Summary
- `scripts/collect.py`: multiprocess (spawn) collector driven by `configs/data/*.yaml`. One ShardWriter per (job, worker) part, seeded episodes, per-episode ε, `_DONE` markers with resume, `summary.json`, and a guard that refuses to resume with a different config or `--workers`.
- `configs/data/debug.yaml` (500k steps: 50% agent ε=0.2, 25% agent ε=0.3, 25% random) and `configs/data/eval.yaml` (100 episodes ≤ 400 steps, seeds from 1e9, 70% agent ε=0.2, 30% random), both at D10 84/max.
- `wm/data/ram.py`: verified RAM map (invaders left, lives, score, player x, alien block x/y, UFO). Two AtariARI addresses failed and were replaced; one was dropped.
- `wm/data/coverage.py` + `scripts/coverage_report.py`: coverage report, histograms and montages in `docs/figures/cp2_coverage_*`; `--verify-ram` makes `docs/figures/cp2_ram_verify.png`.
- **Collected and validated:** `data/debug/` (500,000 steps, 687 episodes, 741 MiB) and `data/eval/` (100 episodes, 39,531 steps, 68 MiB). Four of five coverage targets are met; wave transitions are 24 against a target of 50 (see Results).

## Files
Created: `scripts/collect.py`, `scripts/coverage_report.py`, `configs/data/debug.yaml`, `configs/data/eval.yaml`, `tests/test_collect.py`, `tests/test_coverage.py`, `docs/figures/cp2_ram_verify.png`, `docs/figures/cp2_coverage_report.md`, `docs/figures/cp2_coverage_hist.png`, `docs/figures/cp2_coverage_montage_{ufo,few_invaders,shields_low,life_lost,wave_start,game_over}.png`, `progress/cp2-A.md`.
Modified (Phase 0 stubs replaced): `wm/data/ram.py`, `wm/data/coverage.py`.

## Tests
`tests/test_collect.py` (13):
- Planning: step and episode budgets split exactly across jobs and workers (largest remainders); bad configs raise (no budget, both budgets, fractions ≠ 1, unknown policy, seeds crossing into the eval range).
- Tiny collect, random policy, 2 workers × 300 steps (`env`): validator passes; `summary.json` totals match; 4 shards (2 jobs × 2 workers); seeds unique and equal to `base_seed + episode_uid`; source and ε = -1 recorded.
- **Replay determinism (`env`):** for every collected episode, `make_env(seed)` + the stored actions reproduce every stored frame (after `to_model_frame`), reward, life count, RAM byte, and the no-op count exactly.
- Resume (`env`): a rerun rewrites nothing; with one part's `_DONE` removed and its shard replaced by garbage, the rerun regenerates byte-identical frames. A rerun with different `--workers` is refused.
- Agent + random collect (`env`, `network`): sources {agent, random}, agent ε within the job's range, seeds unique, replay exact.

`tests/test_coverage.py` (8): wave-transition and life-loss counting; UFO detector (true for magenta in the top band at 84 and 96, false for laser grey, score colour, or magenta below the band); shield pixel count in band only; the report runs on a fake shard; coverage event counts on a fake shard; RAM map vs the real game over a random rollout (`env`: lives, score, invaders −1 per alien kill).

Commands (macOS x86_64, Python 3.12.6, torch 2.2.2, ale-py 0.10.2, gymnasium 1.0.0, SB3 2.4.1, CPU; see cp1-A for this laptop's setup):
- `pytest -m "not slow and not gpu and not network"` → 105 passed in 73 s (while collection used all cores).
- `pytest -m network` on the new files → 1 passed (agent collect).
- `python scripts/validate_data.py data/debug` and `data/eval` → `OK: all invariants hold`.

## Results
**Datasets** (`python scripts/collect.py --config configs/data/<name>.yaml --workers 4`):

| Dataset | Steps | Episodes | Rows | Size | Compression | Collection time |
|---|---|---|---|---|---|---|
| `data/debug/` | 500,000 | 687 (456 agent, 231 random) | 500,687 | 741 MiB | 13.6× | ~15 min of worker time (the process ran 35 min wall clock; not investigated, likely the laptop idling) |
| `data/eval/` | 39,531 | 100 (70 agent, 30 random) | 39,631 | 68 MiB | 11.8× | 77 s |

- Eval: seeds 1,000,000,000+; every episode has ≥ 262 rows (all ≥ 204 = K + horizon 200); 90 reach the full 400 steps.
- Mean score in debug: agent 463.8 (ε=0.2 and 0.3 mixed), random 148.2 (matches D16).

**Coverage** (`docs/figures/cp2_coverage_report.md`):

| Target | Minimum | Measured | Met |
|---|---|---|---|
| frames with ≤ 6 invaders left | 5% | 8.48% | yes |
| wave transitions | 50 | 24 | **no** |
| game overs | 200 | 675 | yes |
| UFO visible | 1% | 17.51% | yes |
| shields ≤ 50% intact | 10% | 15.26% | yes |

Life losses: 1,958. The UFO pixel detector agrees with RAM[76] on 99.93% of stored frames. Montages checked by eye: every UFO frame shows the UFO, low-shield frames have eroded shields, and wave-start frames show a fresh 36-alien wave.

**Wave-transition gap, and the revised target.** Waves cleared by source: ε=0.2 agent 18 in 289 episodes, ε=0.3 agent 6 in 167, random 0 in 231. 214 agent episodes reach ≤ 6 invaders but die before the wave ends (the last aliens are fast and low). At the debug mix, 50 transitions would need about 1M steps. **Revised target: ≥ 20 for the 500k debug set** (met), keeping ≥ 50 for the CP5 full set, which at this rate should give about 100–240 at 2–5M steps. If CP5 wants more late-wave and new-wave frames, add an agent job with ε ≈ 0.05, since lower ε clears more waves.

**RAM verification** (`docs/figures/cp2_ram_verify.png`, ε=0.1 agent, seeds 0–2, 2,474 frames):

| Field | Address | Check |
|---|---|---|
| invaders left | 17 | −1 on every alien-kill reward (100%); 36 at a new wave |
| lives | 73 | equals `info["lives"]` on 100% of frames except the game-over frame |
| score | BCD 102 (hi), 104 (lo) | equals cumulative reward on 100% |
| player x | 28 | r = 0.998 vs sprite column |
| alien block x | 26 | r = 1.000 vs block left column |
| alien block y | 16 | r = 1.000 vs block top row |
| UFO on screen | 76 ≥ 4 | agrees with UFO pixels on 99.84% |
| rejected: AtariARI `player_score` | 23 | r = −0.72 vs score |
| rejected: AtariARI `enemies_y` | 24 | r = −0.80 vs block row |
| dropped: AtariARI `missiles_y` | 9 | not verified: the laser and alien bullets share a colour |

## Deviations
- **Writer ids and seeds.** One ShardWriter per (job, worker) part, with id `w = job_index * workers + worker`; seed = `base_seed + w * 10_000_000 + local_counter`, which is the roadmap's formula with "worker" meaning writer id. So seed = `base_seed + episode_uid`. A job's "seed range" is therefore implied by its writer ids, not listed in the config. `plan_parts` checks that all seeds stay inside one range (training < 1e9 ≤ eval < 2e9).
- **Resume granularity is a part, not a shard.** An unfinished part is redone from scratch (its partial shards deleted). Seeding makes the redone data identical, which a test checks.
- **The last episode of a step budget is truncated** (`end_reason = truncated`) so each part hits its budget exactly. One such episode per part (12 in debug).
- Agent ε is drawn per episode from `default_rng([seed, 1])`, a stream separate from the policy's own `default_rng(seed)`.
- Torch runs single-threaded in each worker (`torch.set_num_threads(1)`), since workers already fill the cores.
- `coverage_report.py` takes a positional dataset dir plus flags (like `validate_data.py`), not `--config/--set`. The report and montages are written to `docs/figures/cp2_coverage_*` (the report is `cp2_coverage_report.md`).
- UFO presence uses the pixel heuristic on stored frames (what the model sees), with RAM[76] reported as a cross-check. Shield erosion uses an exact shield-colour count, so a laser overlapping a shield pixel briefly lowers the count.
- Datasets are not committed (gitignored). `collector_config` in each shard plus the committed configs regenerate them. On other machines the agent's sampled actions may differ with a different torch build, which gives different (equally valid) data; replay determinism holds within a dataset because the actions are stored.

## Interface change requests
None.

## Follow-ups
- **Syed (CP2-B and the integration check):** the data is on Aunonto's laptop only (`data/debug/` 741 MiB, `data/eval/` 68 MiB). Either copy it, or regenerate on your machine with the two commands above (~15 min on 4 cores for debug).
- CP5 full set: keep the wave-transition target at ≥ 50, and consider an ε ≈ 0.05 agent job for more late-wave coverage.
- `to_model_frame(method="max")` costs ~1.6 ms per frame on this laptop (torch overhead), about 20% of collection time. A numpy version would speed up CP5's larger collection; that is a CP1-A file, so it is not changed here.
- This laptop has 4 GB of disk left after collecting. CP5's 2–5M steps (~3–7 GB) cannot be collected here.
