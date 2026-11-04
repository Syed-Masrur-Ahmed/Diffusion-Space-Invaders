"""Train the diffusion world model (ROADMAP CP3-B, 4.12, 4.15).

    python scripts/train.py --config configs/train/debug.yaml
    python scripts/train.py --config configs/train/debug.yaml --dry-run --set model.arch=tiny
    python scripts/train.py --config configs/train/debug.yaml --resume auto        # Slurm jobs
    python scripts/train.py --config configs/train/debug.yaml --resume runs/<run_dir>

`--resume auto` continues the most recent runs/<stamp>_<name>/ with a ckpt_last.pt, or starts a
new run if there is none. SIGUSR1 or SIGTERM saves ckpt_last.pt at the next step and exits 0.
"""

import argparse
import sys

from wm.train.trainer import load_run_config, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="key=value override")
    parser.add_argument("--resume", default=None, help='"auto" or a run directory')
    parser.add_argument("--dry-run", action="store_true", help="run 5 steps, write nothing")
    args = parser.parse_args(argv)
    return run(load_run_config(args.config, args.overrides), resume=args.resume, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
