"""Pretrained RL Baselines3 Zoo agent policy (ROADMAP 5, CP1-C)."""

import importlib.util
import sys
import types
import warnings

import numpy as np
import torch

from wm.data.schema import NUM_ACTIONS
from wm.env import AgentObsStack, to_agent_obs

DEFAULT_REPO_ID = "sb3/ppo-SpaceInvadersNoFrameskip-v4"
_FALLBACK_REPO_IDS = (
    "sb3/dqn-SpaceInvadersNoFrameskip-v4",
    "sb3/qrdqn-SpaceInvadersNoFrameskip-v4",
)
# Schedules in the zoo zip are not picklable across SB3 versions (ROADMAP CP1-C).
_LOAD_CUSTOM_OBJECTS = {
    "learning_rate": 0.0,
    "lr_schedule": lambda _: 0.0,
    "clip_range": lambda _: 0.0,
}


def _alias_legacy_gym() -> None:
    """Let the SB3 1.5 zoo zip unpickle, by resolving `gym.spaces` to gymnasium.

    The checkpoint names `gym.spaces.box.Box`. Pointing `sys.modules["gym"]` at the
    gymnasium package loads that file again under a second module name, so the
    unpickled class is not `gymnasium.spaces.Box` and SB3 rejects the space.
    """
    if "gym" in sys.modules or importlib.util.find_spec("gym") is not None:
        return
    import gymnasium.spaces as gspaces

    def put(name: str, **attrs: object) -> types.ModuleType:
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module
        return module

    gym = put("gym")
    spaces = put(
        "gym.spaces",
        Box=gspaces.Box,
        Discrete=gspaces.Discrete,
        Space=gspaces.Space,
    )
    gym.spaces = spaces
    put("gym.spaces.box", Box=gspaces.Box)
    put("gym.spaces.discrete", Discrete=gspaces.Discrete)
    put("gym.spaces.space", Space=gspaces.Space)


def _filename(repo_id: str) -> str:
    return f"{repo_id.rsplit('/', 1)[-1]}.zip"


def _resolve_repo(repo_id: str | None) -> str:
    """Return `repo_id` if it exists, otherwise the first available zoo fallback."""
    from huggingface_hub import repo_exists

    if repo_id is not None:
        if not repo_exists(repo_id):
            raise FileNotFoundError(f"zoo repo {repo_id} does not exist")
        return repo_id
    for candidate in (DEFAULT_REPO_ID, *_FALLBACK_REPO_IDS):
        if repo_exists(candidate):
            return candidate
    raise FileNotFoundError("none of the SB3 zoo Space Invaders repos exist")


def _loader(repo_id: str):
    algo = repo_id.rsplit("/", 1)[-1].split("-", 1)[0]
    if algo == "ppo":
        from stable_baselines3 import PPO

        return PPO.load
    if algo == "dqn":
        from stable_baselines3 import DQN

        return DQN.load
    if algo == "qrdqn":
        from sb3_contrib import QRDQN

        return QRDQN.load
    raise ValueError(f"unsupported zoo algorithm in {repo_id}")


def _load_model(repo_id: str):
    from huggingface_sb3 import load_from_hub

    _alias_legacy_gym()
    path = load_from_hub(repo_id, _filename(repo_id))
    # SB3 < 1.7 saved one shared features_extractor. SB3 2 aliases it to the pi/vf
    # extractors, warns, and still copies the weights. The warning is not a failed load.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="You are probably loading a A2C/PPO model saved with SB3 < 1.7.0",
        )
        return _loader(repo_id)(path, custom_objects=_LOAD_CUSTOM_OBJECTS, device="cpu")


class ZooAgentPolicy:
    """SB3 zoo agent. Observations are 84×84 greyscale stacks, not model frames."""

    def __init__(self, repo_id: str | None = None, model=None) -> None:
        if model is None:
            self.repo_id = _resolve_repo(repo_id)
            model = _load_model(self.repo_id)
        else:
            self.repo_id = repo_id
        self.model = model
        shape = tuple(self.model.observation_space.shape)
        if shape == (4, 84, 84):
            self._channels_first = True
        elif shape == (84, 84, 4):
            self._channels_first = False
        else:
            raise AssertionError(f"expected observation space (4, 84, 84) or (84, 84, 4), got {shape}")
        action_space = getattr(model, "action_space", None)
        if action_space is not None and hasattr(action_space, "n"):
            assert action_space.n == NUM_ACTIONS, action_space
        self._stack = AgentObsStack()
        self._stacked = self._stack.stack
        self._use_reset_frame = False

    def reset(self, seed: int, raw_obs: np.ndarray) -> None:
        """Zero-fill the frame stack with `raw_obs` and seed the agent's sampler."""
        seed_fn = getattr(self.model, "set_random_seed", None)
        if seed_fn is not None:
            seed_fn(seed)
        else:
            torch.manual_seed(seed)
        self._stacked = self._stack.reset(to_agent_obs(raw_obs))
        self._use_reset_frame = True

    def _model_obs(self) -> np.ndarray:
        stacked = self._stacked
        if self._channels_first:
            stacked = np.transpose(stacked, (2, 0, 1))
        return np.ascontiguousarray(stacked)

    def act(self, raw_obs: np.ndarray) -> int:
        """Return a stochastic action. The first call after reset reuses the reset frame."""
        if self._use_reset_frame:
            self._use_reset_frame = False
        else:
            self._stacked = self._stack.push(to_agent_obs(raw_obs))
        action, _state = self.model.predict(self._model_obs(), deterministic=False)
        action = int(np.asarray(action).reshape(-1)[0])
        if not 0 <= action < NUM_ACTIONS:
            raise ValueError(f"agent action {action} is outside 0..{NUM_ACTIONS - 1}")
        return action
