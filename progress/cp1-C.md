# cp1-C: Policies

## Summary
- `RandomPolicy` draws a uniform action and holds it for a 1+Geometric run whose mean is `mean_hold` (default 4).
- `EpsilonMixPolicy` starts such a run with probability ε (default mean hold 2) and otherwise returns the base action. The base policy is stepped on every frame, including during a random hold, so its frame stack stays aligned.
- `ZooAgentPolicy` loads `sb3/ppo-SpaceInvadersNoFrameskip-v4` (DQN, then QR-DQN, if that repo is missing), feeds it an `AgentObsStack` of `to_agent_obs` frames, and samples with `predict(..., deterministic=False)`.
- `scripts/eval_agent.py --episodes 5` prints mean and sample standard deviation for the agent, ε=0.2, ε=0.3, and random, on the same seeds.

## Files
Created:
- `tests/test_policies.py`
- `scripts/eval_agent.py`
- `progress/cp1-C.md`

Modified:
- `wm/agents/policies.py` (`RandomPolicy`, `EpsilonMixPolicy`; `Policy` unchanged)
- `wm/agents/pretrained.py` (`ZooAgentPolicy`)
- `wm/agents/__init__.py` (exports)

## Tests
Added:
- `RandomPolicy` over 10k steps, seed 0: completed hold lengths average 4 ± 0.5, the hold lengths vary, and each action's frequency is within 0.05 of 1/6. The same seed repeats; a different seed does not.
- `EpsilonMixPolicy`: ε=0 matches a deterministic base and does not draw a random action. ε=1 returns the random action, never the base action, while the base still receives every frame. A hold of 3 steps draws once per three steps.
- Frame-stack alignment (fast, fake model): reset stacks the first frame once; the first `act` does not push it again; the next `act` shifts it. Checked for observation shapes `(4, 84, 84)` and `(84, 84, 4)`. `predict` is called with `deterministic=False` and the action is a Python `int`.
- Zoo load and two `act` calls return an int in 0..5 (`network`, `env`).
- Zoo mean score ≥ 2× random over 3 shared seeds (`slow`, `network`, `env`).

Commands (Python 3.13.16, torch 2.14.1, stable-baselines3 2.9.0, gymnasium 1.4.0, ale-py 0.12.1, numpy 2.5.3, macOS arm64, CPU, CUDA unavailable):
- `pytest -m "not slow and not gpu and not network"` → 31 passed in 3.6 s (was 26 at cp0).
- `pytest -m "slow or network or env"` → 6 passed in 8.1 s. That run includes the load test and the 3-episode score test. No `gpu` tests belong to this workflow.

## Results
`python scripts/eval_agent.py --episodes 5` (seeds 0..4, sample std, ddof=1). Repo `sb3/ppo-SpaceInvadersNoFrameskip-v4`, observation space `(4, 84, 84)`.

| policy | mean | std | scores |
|---|---:|---:|---|
| agent | 1041.0 | 263.9 | 1150, 570, 1135, 1170, 1180 |
| epsilon=0.2 | 596.0 | 136.9 | 575, 805, 525, 440, 635 |
| epsilon=0.3 | 291.0 | 179.9 | 115, 425, 305, 105, 505 |
| random | 139.0 | 61.5 | 145, 145, 185, 185, 35 |

No figures. The 3-episode test (seeds 0..2) passed: agent mean ≥ 2× random mean.

## Deviations
- **Legacy `gym` unpickling.** The zoo zip was saved with SB3 1.5.1a8 against `gym` 0.21. `PPO.load` with the roadmap's `custom_objects` and `device="cpu"` fails on Python 3.13 because `gym` is not installed: cloudpickle asks for `gym.spaces.box.Box`. When `gym` is absent, `ZooAgentPolicy` registers that class (and `Discrete`) as the gymnasium classes before load. Assigning `sys.modules["gym"] = gymnasium` does not work: the same file is loaded under a second module name, so the unpickled space is not `isinstance` of `gymnasium.spaces.Box`, and SB3 rejects it. `gym` was not added to `requirements.txt`.
- **SB3 pre-1.7 warning is filtered.** The checkpoint has one shared `features_extractor`. SB3 2 warns that `pi_features_extractor` / `vf_features_extractor` keys are missing, then loads with exact matching off. Those names are aliases of the shared extractor, and the CNN weights match the zip (checked against `policy.pth`). The warning is hidden so the score table stays readable.
- **`load_from_hub` filename** is `{repo_name}.zip` (the name on the hub). The roadmap specifies the repo id, not the filename.
- **Hold length** is `Generator.geometric(1/mean_hold)`. That distribution is the number of trials until the first success, mean `1/p`, which is 1 plus a failures-until-success geometric draw. `mean_hold=4` gives mean 4.
- **First `act` after `reset` does not push.** `reset` already puts the initial frame in the stack (zeros, then that frame), matching `AgentObsStack` / SB3 `VecFrameStack`. Pushing again would duplicate it.
- **`model=` constructor argument** exists so the stack test can run without a download. The default path still loads from the hub.
- **`scripts/eval_agent.py` has no config file.** This workflow does not own `configs/`. `--config` and `--set` are accepted; `--episodes` and `--seed` override them. Defaults: 5 episodes, seed 0, epsilons 0.2 and 0.3, `repo_id` resolved by the policy. Every condition uses seeds `seed .. seed+episodes-1`. The reported std is the sample standard deviation (`ddof=1`).
- **QR-DQN is not a dependency.** The PPO repo exists, so `sb3-contrib` was not appended. The QR-DQN branch imports it only if that repo is the one that loads.
- **Device is `"cpu"`**, as the CP1-C load snippet says, including on a Mac where MPS exists.

## Interface change requests
None.

## Follow-ups
- CP2-A's collector should call `reset(seed, obs)` and then `act` on that same observation. Later `act` calls receive the new frame. A second push of the reset frame would shift the stack one step early.
- The zoo agent was trained on `SpaceInvadersNoFrameskip-v4` with SB3's Atari wrapper (life loss treated as terminal, clipped reward). We feed it max-pooled `ALE/SpaceInvaders-v5` frames. It still scores about 1000, against about 140 for random.
- If the PPO repo disappears, the loader tries DQN, then QR-DQN. QR-DQN needs `sb3-contrib` added to `requirements.txt` at that point.
- A future SB3 that stops accepting the 1.5 shared-extractor state dict will show up as the agent failing the 2× random score test.
