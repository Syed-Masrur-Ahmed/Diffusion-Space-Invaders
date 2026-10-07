"""Keyboard-to-action mapping shared by the play, record, and demo scripts (ROADMAP 4.1)."""

# action -> (move, fire), with move 0=none, 1=right, 2=left
ACTION_TO_MOVE_FIRE = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (2, 0), 4: (1, 1), 5: (2, 1)}
_MOVE_FIRE_TO_ACTION = {mf: a for a, mf in ACTION_TO_MOVE_FIRE.items()}


def keys_to_action(left: bool, right: bool, fire: bool) -> int:
    """Map held keys to an action index; left and right together cancel out."""
    move = 0 if left == right else (2 if left else 1)
    return _MOVE_FIRE_TO_ACTION[(move, int(fire))]


class FireLatch:
    """Keeps fire on for `n` ticks after a key press, so taps shorter than a tick still register."""

    def __init__(self, n: int = 4) -> None:
        self.n = n
        self._left = 0

    def press(self) -> None:
        """Record a fire key-down event."""
        self._left = self.n

    def tick(self, held: bool) -> bool:
        """Return whether to fire this tick, given whether the key is held now."""
        fire = held or self._left > 0
        self._left = max(self._left - 1, 0)
        return fire
