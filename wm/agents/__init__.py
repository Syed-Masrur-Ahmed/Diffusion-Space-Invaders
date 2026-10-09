"""Policies for data collection: random, epsilon mix, and the pretrained zoo agent."""

from wm.agents.policies import EpsilonMixPolicy, Policy, RandomPolicy
from wm.agents.pretrained import ZooAgentPolicy

__all__ = ["EpsilonMixPolicy", "Policy", "RandomPolicy", "ZooAgentPolicy"]
