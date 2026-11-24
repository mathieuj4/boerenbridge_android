"""Opponents that do NOT match the heuristic the bot's default model assumes."""
import random
from agents import HeuristicAgent


class BoldHeuristic(HeuristicAgent):
    """Bids `extra` tricks more than the heuristic's estimate (over-bids systematically)."""
    name = "bold"

    def __init__(self, extra=1.0):
        self.extra = extra

    def estimate(self, v):
        return super().estimate(v) + self.extra


class NoisyHeuristic(HeuristicAgent):
    """Bids like the heuristic, but plays a uniformly random legal card with probability `p`."""
    name = "noisy"

    def __init__(self, p=0.35, seed=0):
        self.p, self.rng = p, random.Random(seed)

    def play(self, v):
        if len(v.legal) > 1 and self.rng.random() < self.p:
            return self.rng.choice(list(v.legal))
        return super().play(v)
