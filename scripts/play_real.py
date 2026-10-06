"""Play Space Invaders through wm.env at agent-step speed, with a flicker-free max-pooled display.

Keys: A/D or arrows move, Space fires, R resets, Esc quits.
"""

import argparse

import numpy as np
import pygame

from wm.controls import FireLatch, keys_to_action
from wm.env import make_env

LEFT_KEYS = (pygame.K_a, pygame.K_LEFT)
RIGHT_KEYS = (pygame.K_d, pygame.K_RIGHT)


def display_array(frame: np.ndarray, scale: int) -> np.ndarray:
    """Upscale an (H,W,3) frame by `scale` (x2 extra horizontally) and transpose to pygame's (W,H,3)."""
    # Atari pixels are wide, so widen for display only; the env frame itself is never altered.
    return frame.repeat(scale, axis=0).repeat(2 * scale, axis=1).swapaxes(0, 1)


def run(seed: int = 0, fps: float = 15, scale: int = 3, max_ticks: int | None = None) -> int:
    """Run the game loop until Esc, window close, or `max_ticks`; return the number of ticks run."""
    pygame.init()
    env = make_env(seed)
    obs, info = env.reset()
    screen = pygame.display.set_mode((obs.shape[1] * 2 * scale, obs.shape[0] * scale))
    clock = pygame.time.Clock()
    latch = FireLatch()
    score, lives, game_over = 0.0, info["lives"], False

    ticks = 0
    running = True
    while running and (max_ticks is None or ticks < max_ticks):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                latch.press()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                score, lives, game_over = 0.0, info["lives"], False

        keys = pygame.key.get_pressed()
        fire = latch.tick(keys[pygame.K_SPACE])
        if not game_over:
            left = any(keys[k] for k in LEFT_KEYS)
            right = any(keys[k] for k in RIGHT_KEYS)
            obs, reward, terminated, truncated, info = env.step(keys_to_action(left, right, fire))
            score += reward
            lives = info["lives"]
            game_over = terminated or truncated

        pygame.surfarray.blit_array(screen, display_array(obs, scale))
        status = " | GAME OVER: press R" if game_over else ""
        pygame.display.set_caption(f"Space Invaders | score {score:.0f} | lives {lives}{status}")
        pygame.display.flip()
        clock.tick(fps)
        ticks += 1

    env.close()
    pygame.quit()
    return ticks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fps", type=float, default=15, help="agent steps per second (15 = real speed)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=int, default=3, help="integer display upscaling")
    args = parser.parse_args()
    run(seed=args.seed, fps=args.fps, scale=args.scale)


if __name__ == "__main__":
    main()
