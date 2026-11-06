# Diffusion-Space-Invaders

An action-conditioned diffusion world model of Atari Space Invaders. See `ROADMAP.md` for the plan
and `CLAUDE.md` for how to work in this repo.

## Milestone 1

Milestone 1 compares a baseline, **copy-last-frame** (0 parameters), with a simple model, the
**`small` EDM U-Net** (14,263,683 parameters, 20k training steps), on the same 64 held-out val
windows. Everything is in `notebooks/milestone1.ipynb`; saved results are in `results/milestone1/`.

**1. Get the code**

```bash
git clone https://github.com/Syed-Masrur-Ahmed/Diffusion-Space-Invaders.git
cd Diffusion-Space-Invaders
git checkout milestone1
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

**3. Download the checkpoint** (224 MB) from [Google Drive](https://drive.google.com/file/d/1bpBC0tTF5uZGqE2Mz3wMtQYqJqwFfAKY/view?usp=sharing) and save it as
`runs/20261105-0558_debug_small/checkpoints/ckpt_last.pt`:

```bash
mkdir -p runs/20261105-0558_debug_small/checkpoints
# move the downloaded file to runs/20261105-0558_debug_small/checkpoints/ckpt_last.pt
```

**4. Get the data** (`data/debug/`, 740 MB, 12 HDF5 shards) from [Google Drive](https://drive.google.com/file/d/1SRgxvCyY3Drwo1DSyvv34Q1z6UBD5LgD/view?usp=drive_link) and unpack it so
the shards are at `data/debug/shard_w*.h5`. Or regenerate it (about 10 minutes; downloads the
agent policy from the Hugging Face Hub):

```bash
python scripts/collect.py --config configs/data/debug.yaml --workers 4
```

Keep `--workers 4`: the shard names, and so the train/val split, depend on it. A regenerated
dataset can still differ slightly from ours (for example with other library versions or
hardware), so the numbers may not match `results/milestone1/` exactly and the notebook's
match check can fail; the saved results are the reference.

**5. Run the notebook**

```bash
jupyter notebook notebooks/milestone1.ipynb     # or, headless:
jupyter nbconvert --to notebook --execute --inplace notebooks/milestone1.ipynb
```

With `RUN_EVAL = True` (the default) the notebook re-evaluates both models and checks that the
numbers match the saved `results/milestone1/eval/*/metrics.json`. Measured on an Apple M3 laptop:
25 s with MPS, 33 s on CPU (the notebook picks CUDA, then MPS, then CPU); expect around a minute on
an older laptop CPU, and well under a minute on a CUDA GPU (not measured). Section 6 (memorization
check) adds about 3 minutes on MPS or a GPU, where it fine-tunes the model on 20 training windows;
on CPU it shows the saved run from `results/milestone1/memorize/` instead (set `MEMORIZE = True`
to force it, 15 to 30 minutes). The first run also
downloads the AlexNet weights for LPIPS (233 MB, needs internet).

To run **without the data or the checkpoint**, set `RUN_EVAL = False` in the setup cell: the
notebook then shows the saved metrics, grids, and training curve from `results/milestone1/`.

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
