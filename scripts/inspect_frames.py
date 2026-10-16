"""Inspect processed frames for the D10 resolution decision (ROADMAP CP1-A).

Default: write a figure of one raw frame and its {84, 96} x {area, max} x {RGB, grey} versions
(4x nearest upscaling), with zoomed crops around the player's laser.
--laser-stats: print how often the laser stays visible in each version during firing rollouts.
"""

import argparse
from collections.abc import Iterator
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: only saves files
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from wm.data.schema import ACTION_MEANINGS  # noqa: E402
from wm.env import make_env, to_model_frame  # noqa: E402

# Calibrated once by frame differencing right after FIRE (ALE 0.10, see progress/cp1-A.md):
# the player laser and the alien bullets share this colour, but the game draws the player
# laser only on even episode frame numbers (always moving up) and alien bullets only on odd
# ones (always moving down). That alternation is the flicker that max-pooling removes.
LASER_RGB = np.array([142, 142, 142], np.uint8)
LASER_PARITY = 0
SHIELD_TOP = 157  # first raw row of the shields; the laser is measured above it and below the aliens
VISIBLE_DIFF = 40  # min max-channel difference from the black background to count as visible
UPSCALE = 4

SIZES = (84, 96)
METHODS = ("area", "max")
FIRE_ACTIONS = [ACTION_MEANINGS.index(a) for a in ("FIRE", "RIGHTFIRE", "LEFTFIRE")]


def to_grey(frame: np.ndarray) -> np.ndarray:
    """Luma greyscale (ROADMAP 4.5) of an RGB uint8 frame, as uint8 (H,W)."""
    luma = frame.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    return np.round(luma).astype(np.uint8)


def versions(raw: np.ndarray) -> dict[str, np.ndarray]:
    """Every candidate model frame of a raw frame, keyed like '84 area RGB'."""
    out = {}
    for size in SIZES:
        for method in METHODS:
            rgb = to_model_frame(raw, size, method)
            out[f"{size} {method} RGB"] = rgb
            out[f"{size} {method} grey"] = to_grey(rgb)
    return out


def firing_rollout(seed: int, steps: int) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
    """Yield (max-pooled obs, last raw frame, its episode frame number) for a seeded firing rollout."""
    env = make_env(seed)
    env.reset()
    ale = env.unwrapped.ale
    rng = np.random.default_rng(seed)
    hold = 0
    for _ in range(steps):
        if hold == 0:  # fire while drifting left and right, so the laser leaves from many columns
            action, hold = int(rng.choice(FIRE_ACTIONS)), int(rng.integers(1, 9))
        hold -= 1
        obs, _, terminated, truncated, _ = env.step(action)
        yield obs, ale.getScreenRGB(), ale.getEpisodeFrameNumber()
        if terminated or truncated:
            env.reset()
    env.close()


def laser_mask(obs: np.ndarray, last_raw: np.ndarray, frame_number: int) -> np.ndarray:
    """Player-laser pixels of the max-pooled obs that lie between the lowest alien and the shields."""
    in_last = (last_raw == LASER_RGB).all(-1)
    if frame_number % 2 == LASER_PARITY:
        mask = in_last
    else:  # the laser is in the previous raw frame; laser-coloured pixels of the last one are alien bullets
        mask = (obs == LASER_RGB).all(-1) & ~in_last

    other = (obs.max(-1) > 0) & ~(obs == LASER_RGB).all(-1)
    alien_rows = np.flatnonzero(other[:SHIELD_TOP].any(-1))
    lowest_alien = alien_rows[-1] if len(alien_rows) else -1
    mask[: lowest_alien + 1] = False
    mask[SHIELD_TOP:] = False
    return mask


def _region(frame: np.ndarray, box: tuple[int, int, int, int], raw_shape: tuple[int, int]) -> np.ndarray:
    """The part of `frame` covering the raw-pixel box (y0, y1, x0, x1), inclusive."""
    y0, y1, x0, x1 = box
    sy, sx = frame.shape[0] / raw_shape[0], frame.shape[1] / raw_shape[1]
    return frame[int(y0 * sy) : int(y1 * sy) + 1, int(x0 * sx) : int(x1 * sx) + 1]


def is_visible(region: np.ndarray) -> bool:
    """A pixel in `region` differs from the black background by >= VISIBLE_DIFF in some channel."""
    return int(region.max()) >= VISIBLE_DIFF  # background is 0, so the difference is the value


def laser_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding box (y0, y1, x0, x1) of the mask, or None if it is empty."""
    ys, xs = np.nonzero(mask)
    return (ys.min(), ys.max(), xs.min(), xs.max()) if len(ys) else None


def laser_stats(seeds: list[int], steps: int) -> dict[str, tuple[int, int]]:
    """For each version, (frames with the laser visible, frames with the laser in the play area)."""
    counts: dict[str, list[int]] = {}
    for seed in seeds:
        for obs, last_raw, frame_number in firing_rollout(seed, steps):
            box = laser_box(laser_mask(obs, last_raw, frame_number))
            if box is None:
                continue
            frames = {"raw max-pooled": obs, "raw last frame": last_raw, **versions(obs)}
            for name, frame in frames.items():
                c = counts.setdefault(name, [0, 0])
                c[0] += is_visible(_region(frame, box, obs.shape[:2]))
                c[1] += 1
    return {name: (v, n) for name, (v, n) in counts.items()}


def _show(ax: plt.Axes, image: np.ndarray, title: str) -> None:
    big = image.repeat(UPSCALE, axis=0).repeat(UPSCALE, axis=1)
    ax.imshow(big, cmap="gray" if big.ndim == 2 else None, vmin=0, vmax=255, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def make_figure(seed: int, steps: int, out: Path) -> None:
    """Save the raw-vs-versions grid for the first rollout frame with a laser mid-flight."""
    for obs, last_raw, frame_number in firing_rollout(seed, steps):
        box = laser_box(laser_mask(obs, last_raw, frame_number))
        if box is not None and box[1] - box[0] >= 2:  # a full laser segment, not one clipped pixel
            break
    else:
        raise RuntimeError(f"no laser found in {steps} steps of seed {seed}")

    cy, cx = (box[0] + box[1]) // 2, (box[2] + box[3]) // 2
    crop_box = (max(cy - 20, 0), min(cy + 20, 209), max(cx - 16, 0), min(cx + 16, 159))
    frames = {"raw (210x160)": obs, **versions(obs)}

    fig, axes = plt.subplots(2, len(frames), figsize=(2.2 * len(frames), 7))
    for col, (name, frame) in enumerate(frames.items()):
        _show(axes[0, col], frame, name)
        _show(axes[1, col], _region(frame, crop_box, obs.shape[:2]), f"{name}\nlaser crop")
    fig.suptitle(f"seed {seed}: raw max-pooled frame and model-frame candidates ({UPSCALE}x nearest)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--laser-stats", action="store_true", help="print laser visibility rates instead of the figure")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 2],
                        help="rollout seeds; 0 and 2 have even and odd no-op counts, covering both flicker phases")
    parser.add_argument("--steps", type=int, default=300, help="agent steps per rollout")
    parser.add_argument("--out", type=Path, default=Path("docs/figures/cp1_frames.png"))
    args = parser.parse_args()

    if not args.laser_stats:
        make_figure(args.seeds[0], args.steps, args.out)
        return
    print(f"laser visibility, seeds {args.seeds}, {args.steps} steps each (visible >= {VISIBLE_DIFF} above black)")
    for name, (visible, total) in laser_stats(args.seeds, args.steps).items():
        print(f"  {name:16s} {visible:4d}/{total:<4d} {100 * visible / max(total, 1):6.1f}%")


if __name__ == "__main__":
    main()
