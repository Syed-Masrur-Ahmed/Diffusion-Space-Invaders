#!/usr/bin/env bash
set -e

# pygame 2.6.1 has macOS wheels through Python 3.13 only. `python3` here is 3.14,
# which forces a source build that fails without SDL2 headers.
if ! command -v python3.13 >/dev/null 2>&1; then
  echo "python3.13 is required (pygame has no Python 3.14 wheels)." >&2
  exit 1
fi

# --clear replaces a leftover 3.14 venv instead of reusing it.
python3.13 -m venv --clear .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
# Makes `wm` importable when running scripts/*.py.
.venv/bin/pip install -e .
