"""Plain Gymnasium player: every emulator frame, no max-pooling (reference for scripts/play_real.py)."""

import numpy as np
import gymnasium as gym
import ale_py
from gymnasium.utils.play import play

gym.register_envs(ale_py)


class WideRender(gym.Wrapper):
    """Doubles the rendered width so Atari's wide pixels look right. Display only."""

    def render(self):
        return np.repeat(self.env.render(), 2, axis=1)


env = gym.make(
    "ALE/SpaceInvaders-v5",
    render_mode="rgb_array",        # play() draws the frames itself via pygame
    frameskip=1,                    # every emulator frame, so human play feels normal
    repeat_action_probability=0.0,  # sticky actions off, matching the project setup
)
env = WideRender(env)               # cosmetic aspect-ratio fix; delete this line to undo

keys_to_action = {
    (" ",): 1,       # FIRE
    ("d",): 2,       # RIGHT
    ("a",): 3,       # LEFT
    ("d", " "): 4,   # RIGHTFIRE
    ("a", " "): 5,   # LEFTFIRE
}

play(env, keys_to_action=keys_to_action, noop=0, zoom=2, fps=60)
