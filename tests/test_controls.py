import itertools

from wm.controls import ACTION_TO_MOVE_FIRE, FireLatch, keys_to_action
from wm.data.schema import ACTION_MEANINGS, NUM_ACTIONS


EXPECTED = {
    # (left, right, fire): action name
    (False, False, False): "NOOP",
    (False, False, True): "FIRE",
    (False, True, False): "RIGHT",
    (True, False, False): "LEFT",
    (False, True, True): "RIGHTFIRE",
    (True, False, True): "LEFTFIRE",
    (True, True, False): "NOOP",  # left and right cancel
    (True, True, True): "FIRE",
}


def test_keys_to_action_all_combinations():
    for keys in itertools.product([False, True], repeat=3):
        assert ACTION_MEANINGS[keys_to_action(*keys)] == EXPECTED[keys]


def test_action_to_move_fire_agrees_with_keys():
    assert sorted(ACTION_TO_MOVE_FIRE) == list(range(NUM_ACTIONS))
    for action, (move, fire) in ACTION_TO_MOVE_FIRE.items():
        left, right = move == 2, move == 1
        assert keys_to_action(left, right, bool(fire)) == action


def test_fire_latch_holds_tap_for_n_ticks():
    latch = FireLatch(n=4)
    latch.press()
    assert [latch.tick(held=False) for _ in range(6)] == [True] * 4 + [False] * 2


def test_fire_latch_held_key_keeps_firing():
    latch = FireLatch(n=2)
    latch.press()
    assert all(latch.tick(held=True) for _ in range(10))
    assert latch.tick(held=False) is False


def test_fire_latch_idle_never_fires():
    latch = FireLatch()
    assert not any(latch.tick(held=False) for _ in range(5))
