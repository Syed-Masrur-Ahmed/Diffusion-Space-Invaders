# CLAUDE.md

An action-conditioned diffusion world model of Atari Space Invaders (team Radiant Descent).
**`ROADMAP.md` is the source of truth.** This file says how to work. The roadmap says what to build.

## Before you start

1. Read `ROADMAP.md` sections 0, 2, 3, and 4, then your workflow's section in section 5.
2. Check you are on the right branch (`git branch --show-current`). It should be `cpN/X-<name>`, created from tag `cp(N-1)`.
3. Read `progress/` notes from the previous checkpoint that relate to your workflow.
4. Before writing code, state a short plan: the files you will touch, the tests you will add, and the order. Keep it to a few lines.

## Scope rules

- Edit only the files your workflow **Owns** (listed in its roadmap section). If another file needs a change, stop and say so.
- Never change a `[FROZEN]` interface in section 4. If one seems wrong, write it under "Interface change requests" in your progress note and keep going against the current interface.
- Don't edit `ROADMAP.md` on a workflow branch. Only the integrator edits it, at checkpoints.
- `requirements.txt`: append lines only. Add a dependency only if the roadmap names it or there's no reasonable way around it, and say why in the progress note.
- Never commit data, checkpoints, or run outputs (`data/`, `runs/`, `*.h5`, `*.pt`).
- If the roadmap is ambiguous, pick the simplest option consistent with section 4 and record it under "Deviations". If a requirement conflicts with section 4, stop and ask.

## How to make changes

- **Small steps.** Make one logical change at a time: one function, one class, or one test file. Run the tests, then move on.
- **Read before editing.** Open the file and its tests before changing it. Don't rewrite working code that's outside your task.
- **Tests first.** Write the tests your workflow lists, watch them fail, then implement until they pass.
- **Keep diffs reviewable.** If a change is growing past ~200 lines, split it into separate commits.
- **Don't guess APIs.** For gymnasium, ale-py, SB3, h5py, or torch behaviour you're unsure of, check the installed version (`pip show`, `help()`, a small script) instead of relying on memory.

## Testing

```bash
source .venv/bin/activate
pytest -m "not slow and not gpu and not network"   # fast suite: run after every change and before every commit
pytest tests/test_<module>.py -x                    # while iterating on one module
pytest -m "slow or network or env"                  # before finishing, if the machine supports it
```

- The fast suite must pass before every commit and finish in under 3 minutes on a laptop CPU.
- Tests use the tiny fixtures in `tests/fixtures.py` (`make_fake_shard`, `TinyDenoiserNet`). Don't make the fast suite depend on real datasets, downloads, or a GPU.
- Mark tests correctly: `slow` (>30 s), `gpu`, `network`, `env` (needs ALE).
- For any code that indexes frames or actions, add an alignment test using `encode_rows=True` fake shards. Off-by-one errors are this project's most likely silent bug.
- Never weaken, skip, or delete a test to make it pass. If a test looks wrong, explain why and ask.

## Code style

- **Minimal.** Write the least code that meets the spec. No speculative options, no abstractions for one caller, no frameworks the roadmap doesn't ask for.
- **Clean.** Use clear names, type hints on public functions, and a one-line docstring for each public function or class. Comment only on non-obvious reasoning, such as why a max-pool or a dtype cast is there.
- Delete dead code instead of commenting it out. Don't leave debug prints or stray notebooks behind.
- Keep functions short and pure where possible. Put I/O, seeding, and device choice at the edges, in scripts.
- Shared constants (action meanings, `NULL_ACTION`, dataset field names) come from `wm/data/schema.py`. Never hardcode them anywhere else.
- Every script takes `--config` and `--set key=value` (section 4.12). Don't hardcode paths, hyperparameters, or cluster settings.
- Code must run on CPU. Choose the device with `"cuda" if torch.cuda.is_available() else "cpu"`.
- Training and evaluation code must never import pygame (it runs headless on Discovery).

## Project pitfalls (full list in ROADMAP 4.14)

- `actions[i]` is taken at frame `i` and produces frame `i+1`.
- Windows never cross episode boundaries. The train/val split is by episode, never by frame.
- `cv2.resize` takes `(W, H)`. Save RGB images with `imageio`, not `cv2.imwrite`.
- Cast `uint8` to a wider type before doing arithmetic on frames.
- Open HDF5 files lazily inside each DataLoader worker.
- Never aspect-correct or otherwise alter stored frames. Widening is only for display.
- Sample with EMA weights, `eval()`, and `inference_mode`. CFG with `w=1` must not double the batch.

## Cluster (Discovery)

Only Syed has Discovery access. If your workflow needs a GPU run, make it runnable through
`scripts/slurm/*.sbatch` and put the config path and the exact `sbatch` command in your
progress note so Syed can submit it.

## Finishing a workflow

1. The fast suite passes. Run the workflow's slow, network, and gpu tests if possible, and report the results.
2. Fill in `progress/cpN-X.md` using the template in ROADMAP section 3.5, including numbers, figure paths, deviations, and follow-ups.
3. Commit with the prefix `[cpN-X]`, in small commits with clear messages.
4. Report back briefly: what exists now, the test results, and anything the integrator or another track needs to know.