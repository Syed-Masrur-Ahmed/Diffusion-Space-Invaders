"""Inverse dynamics model: which action turned f_t into f_{t+1} (ROADMAP CP5-A).

Two heads, move (0 none, 1 right, 2 left) and fire (0/1), from `ACTION_TO_MOVE_FIRE`. Fire is
often unobservable (a laser already in flight makes FIRE a no-op), so its accuracy on real val
is a ceiling, not a target. Frames are (B,C,H,W) in [-1,1], as in the training sample contract.
"""

from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from wm.controls import ACTION_TO_MOVE_FIRE
from wm.data.schema import NUM_ACTIONS

_MOVE_FIRE = torch.tensor([ACTION_TO_MOVE_FIRE[a] for a in range(NUM_ACTIONS)])  # (NUM_ACTIONS, 2)


def action_targets(actions: Tensor) -> tuple[Tensor, Tensor]:
    """(move, fire) class indices for action ids of any shape."""
    mf = _MOVE_FIRE.to(actions.device)[actions]
    return mf[..., 0], mf[..., 1]


class IDM(nn.Module):
    """Small CNN on concat(f_t, f_{t+1}) with a move head (3) and a fire head (2)."""

    def __init__(self, C: int = 3, H: int = 84, W: int = 84, ch: tuple[int, ...] = (32, 64, 64, 128),
                 hidden: int = 256) -> None:
        super().__init__()
        self.config = {"C": C, "H": H, "W": W, "ch": list(ch), "hidden": hidden}
        # The player moves ~0.5 px per step at 84, so the first conv keeps full resolution.
        layers, c_in = [nn.Conv2d(2 * C, ch[0], 3, padding=1), nn.SiLU()], ch[0]
        for c in ch:
            layers += [nn.Conv2d(c_in, c, 3, stride=2, padding=1), nn.SiLU()]
            c_in = c
        self.features = nn.Sequential(*layers, nn.Flatten())
        n_features = self.features(torch.zeros(1, 2 * C, H, W)).shape[1]
        self.trunk = nn.Sequential(nn.Linear(n_features, hidden), nn.SiLU())
        self.move = nn.Linear(hidden, 3)
        self.fire = nn.Linear(hidden, 2)

    def forward(self, frame: Tensor, next_frame: Tensor) -> tuple[Tensor, Tensor]:
        """frame, next_frame (B,C,H,W) in [-1,1] -> move logits (B,3), fire logits (B,2)."""
        h = self.trunk(self.features(torch.cat([frame, next_frame], dim=1)))
        return self.move(h), self.fire(h)


def loss_and_accuracy(model: IDM, frame: Tensor, next_frame: Tensor, actions: Tensor) -> tuple[Tensor, dict[str, float]]:
    """Summed cross-entropy of both heads, and the batch accuracy of each."""
    move_t, fire_t = action_targets(actions)
    move, fire = model(frame, next_frame)
    loss = F.cross_entropy(move, move_t) + F.cross_entropy(fire, fire_t)
    return loss, {"move": (move.argmax(1) == move_t).float().mean().item(),
                  "fire": (fire.argmax(1) == fire_t).float().mean().item()}


def save_idm(path: str | Path, model: IDM, **extra) -> None:
    """Save weights and constructor config (plus any `extra` entries, e.g. step and metrics)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": model.config, **extra}, path)


def load_idm(path: str | Path, device: str | torch.device = "cpu") -> IDM:
    """Load an IDM saved by `save_idm`, in eval mode."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = IDM(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval()
