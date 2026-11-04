"""Record human Space Invaders play into HDF5 shards (ROADMAP 4.4, CP3-A).

    python scripts/play_record.py --config configs/data/human.yaml [--hz 15] [--scale 3]

Keys: A/D or arrows move, Space fires, Esc ends the episode (end_reason=stopped) and quits.
After a game over, R starts a new episode. One agent step per tick, so --hz 15 is real speed;
lower is easier and records the same game. Each session writes as its own worker id (the
smallest unused one >= first_worker), with episode seeds HUMAN_SEED_BASE + episode_uid.
"""

import argparse
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pygame

from wm.config import load_config
from wm.controls import FireLatch, keys_to_action
from wm.data.schema import (END_STOPPED, END_TERMINATED, END_TRUNCATED, EPISODE_UID_STRIDE,
                            HUMAN_SEED_BASE, SOURCE_HUMAN)
from wm.data.writer import ShardWriter
from wm.env import make_env, to_model_frame

MAX_HZ = 15  # real game speed: 15 agent steps of 4 emulator frames = 60 frames per second
LEFT_KEYS = (pygame.K_a, pygame.K_LEFT)
RIGHT_KEYS = (pygame.K_d, pygame.K_RIGHT)


def next_worker(out_dir: str | Path, first_worker: int) -> int:
    """Smallest worker id >= first_worker that no shard in `out_dir` uses yet."""
    used = {int(m.group(1)) for p in Path(out_dir).glob("shard_w*_*.h5")
            if (m := re.fullmatch(r"shard_w(\d+)_\d+\.h5", p.name))}
    worker = first_worker
    while worker in used:
        worker += 1
    return worker


def display_array(frame: np.ndarray, scale: int) -> np.ndarray:
    """Upscale an (H,W,3) frame by `scale` (x2 extra horizontally) and transpose to pygame's (W,H,3)."""
    # Atari pixels are wide, so widen for display only; recorded frames are never altered.
    return frame.repeat(scale, axis=0).repeat(2 * scale, axis=1).swapaxes(0, 1)


def run(cfg: dict, hz: float = MAX_HZ, scale: int = 3, max_ticks: int | None = None,
        get_events: Callable[[], list] = pygame.event.get) -> dict:
    """Record until Esc, window close, or `max_ticks`; returns {"worker", "episodes", "steps"}."""
    size, method = int(cfg["resolution"]), cfg["resize_method"]
    worker = next_worker(cfg["out_dir"], int(cfg["first_worker"]))
    attrs = {"env_id": "ALE/SpaceInvaders-v5", "frameskip": 4, "maxpool": True, "resolution": size,
             "resize_method": method, "collector_config": {**cfg, "recorder": "play_record", "hz": hz}}
    pygame.init()
    writer = ShardWriter(cfg["out_dir"], worker, (size, size, 3), attrs, int(cfg["max_rows"]))
    clock = pygame.time.Clock()
    latch, held = FireLatch(), set()
    episodes = steps = 0
    env = screen = None

    def start_episode() -> tuple[np.ndarray, dict]:
        nonlocal env
        if env is not None:
            env.close()
        seed = HUMAN_SEED_BASE + worker * EPISODE_UID_STRIDE + episodes  # = HUMAN_SEED_BASE + episode_uid
        env = make_env(seed, max_episode_steps=int(cfg["max_episode_steps"]))
        obs, info = env.reset()
        writer.begin_episode(to_model_frame(obs, size, method), env.unwrapped.ale.getRAM(), info["lives"],
                             seed, info["noops"], SOURCE_HUMAN, -1.0)
        return obs, info

    obs, info = start_episode()
    screen = pygame.display.set_mode((obs.shape[1] * 2 * scale, obs.shape[0] * scale))
    score, lives, in_episode, ticks = 0.0, info["lives"], True, 0
    while max_ticks is None or ticks < max_ticks:
        quit_ = False
        for event in get_events():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                quit_ = True
            elif event.type == pygame.KEYDOWN:
                held.add(event.key)
                if event.key == pygame.K_SPACE:
                    latch.press()
                elif event.key == pygame.K_r and not in_episode:
                    obs, info = start_episode()
                    score, lives, in_episode = 0.0, info["lives"], True
            elif event.type == pygame.KEYUP:
                held.discard(event.key)
            elif event.type == pygame.WINDOWFOCUSLOST:
                held.clear()  # key-ups are not delivered to an unfocused window
        if quit_:
            break

        fire = latch.tick(pygame.K_SPACE in held)
        if in_episode:
            action = keys_to_action(any(k in held for k in LEFT_KEYS), any(k in held for k in RIGHT_KEYS), fire)
            obs, reward, terminated, truncated, info = env.step(action)
            writer.add_step(action, float(reward), to_model_frame(obs, size, method),
                            env.unwrapped.ale.getRAM(), info["lives"])
            steps += 1
            score += reward
            lives = info["lives"]
            if terminated or truncated:
                writer.end_episode(END_TERMINATED if terminated else END_TRUNCATED, terminated)
                episodes += 1
                in_episode = False

        pygame.surfarray.blit_array(screen, display_array(obs, scale))
        status = "" if in_episode else " | GAME OVER: R new game, Esc quits"
        pygame.display.set_caption(f"REC w{worker:03d} ep {episodes} | score {score:.0f} | lives {lives}{status}")
        pygame.display.flip()
        if hz > 0:
            clock.tick(hz)
        ticks += 1

    if in_episode:
        writer.end_episode(END_STOPPED, terminated=False)
        episodes += 1
    writer.close()
    env.close()
    pygame.quit()
    return {"worker": worker, "episodes": episodes, "steps": steps}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="key=value override")
    parser.add_argument("--hz", type=float, default=MAX_HZ, help=f"agent steps per second, at most {MAX_HZ}")
    parser.add_argument("--scale", type=int, default=3, help="integer display upscaling")
    args = parser.parse_args(argv)
    if not 0 < args.hz <= MAX_HZ:
        parser.error(f"--hz must be in (0, {MAX_HZ}]")
    cfg = load_config(args.config, args.overrides)
    stats = run(cfg, hz=args.hz, scale=args.scale)
    print(f"{cfg['out_dir']}: worker {stats['worker']}, {stats['episodes']} episodes, {stats['steps']} steps. "
          f"Check with: python scripts/validate_data.py {cfg['out_dir']}")


if __name__ == "__main__":
    main()
