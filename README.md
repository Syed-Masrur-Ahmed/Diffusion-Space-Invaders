# Diffusion-Space-Invaders

An action-conditioned diffusion world model of Atari Space Invaders. See `ROADMAP.md` for the plan
and `CLAUDE.md` for how to work in this repo.

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
