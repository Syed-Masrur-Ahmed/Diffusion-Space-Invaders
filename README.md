# Diffusion-Space-Invaders

An action-conditioned diffusion world model of Atari Space Invaders. See `ROADMAP.md` for the plan
and `CLAUDE.md` for how to work in this repo.

## Milestone 2

Milestone 2 moves to the full 5M-step dataset, checks the training pipeline (E1-E5), runs
one-change experiments (E6-E8), evaluates once on a new 100-episode test set, and analyses the
errors (E9). Everything is in `notebooks/radiant_descent_milestone2.ipynb`; saved results are in
`results/milestone2/`.

**1. Get the code**

```bash
git clone https://github.com/Syed-Masrur-Ahmed/Diffusion-Space-Invaders.git
cd Diffusion-Space-Invaders
git checkout milestone2
```

**2. Install.** Use **Python 3.13**. `pyproject.toml` allows 3.13 and newer, but pygame has no
Python 3.14 wheels on macOS, so 3.13 is the supported version. Either:

```bash
bash setup.sh && source .venv/bin/activate                              # needs python3.13 on PATH
# or, with uv:
uv venv --python 3.13 .venv && source .venv/bin/activate && uv pip install -r requirements.txt
```

Then install PyTorch for your machine with the command from
[pytorch.org](https://pytorch.org/get-started/locally/) (the default build on a laptop, the CUDA
build on a GPU machine).

**3. Run the notebook.** No data, checkpoints or GPU are needed:

```bash
jupyter notebook notebooks/radiant_descent_milestone2.ipynb     # or, headless:
jupyter nbconvert --to notebook --execute --inplace notebooks/radiant_descent_milestone2.ipynb
```

Restart and Run All takes a few seconds on a laptop (about 3 s on an Apple laptop CPU). The
training runs and the long evaluations (full dataset, test set, rollouts, counterfactuals, E9)
ran on a GPU cluster, and the notebook loads their saved results from `results/milestone2/`.
They can be re-run with `scripts/slurm/m2_runs.sh` (see the comments at its top), which needs
the full dataset and a Slurm cluster.

**4. Optional: re-run the pipeline checks.** E1-E3 can be recomputed locally. They need the
500k-step debug set (`data/debug/`, 740 MB) from
[Google Drive](https://drive.google.com/file/d/1SRgxvCyY3Drwo1DSyvv34Q1z6UBD5LgD/view?usp=drive_link),
unpacked so the shards are at `data/debug/shard_w*.h5`, or regenerated with
`python scripts/collect.py --config configs/data/debug.yaml --workers 4` (about 10 minutes).
Then set `RUN_CHECKS = True` in the setup cell. They are slow on a laptop: E2 takes about 7 minutes
on CPU and E3 about 1 hour on Apple MPS.

## Setup

Requires `python3.13` (pygame has no Python 3.14 wheels on macOS).

**macOS:** keep the repo outside iCloud-synced folders (`~/Documents`, `~/Desktop` when
"Desktop & Documents" sync is on). iCloud sets the hidden flag on files in `.venv`, Python 3.13
then skips the `.pth` file that makes `wm` importable, and scripts fail with
`No module named 'wm'`.

```bash
bash setup.sh                  # creates .venv and installs requirements.txt (including the wm package)
source .venv/bin/activate
```

Then install PyTorch (`torch>=2.2`) with the command for your machine from
[pytorch.org](https://pytorch.org/get-started/locally/): the CUDA build on the cluster, the
default build on a laptop. Everything runs on CPU when no GPU is present, just slowly.

## Play

```bash
python scripts/play_real.py            # project env: 4-frame skip, max-pooled (flicker-free) frames
python scripts/play_real.py --fps 8    # slower game, easier play (15 = real speed)
python scripts/play_gym.py             # plain Gymnasium player, every emulator frame
```

`play_real.py` keys: A/D or arrows to move, Space to fire, R to reset, Esc to quit.
Options: `--fps` (agent steps per second), `--seed`, `--scale` (display size).
`play_gym.py` keys: A/D to move, Space to fire.

## Tests

```bash
pytest -m "not slow and not gpu and not network"   # fast suite, run before every commit
pytest -m "slow or network or env"                 # longer tests, where the machine supports them
```
