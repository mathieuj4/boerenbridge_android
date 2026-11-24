"""Baseline agents: a random one and a simple hand-evaluating heuristic."""
from __future__ import annotations
import random
from boerenbridge import Agent, BidView, PlayView, beats, trick_winner


class RandomAgent(Agent):
    name = "random"
    def __init__(self, rng): self.rng = rng

    def bid(self, v: BidView) -> int:
        options = [b for b in range(v.n_cards + 1) if b != v.forbidden]
        return self.rng.choice(options)

    def play(self, v: PlayView):
        return self.rng.choice(v.legal)


TRUMP_P = {14: .92, 13: .78, 12: .62, 11: .48, 10: .38, 9: .30, 8: .24, 7: .19, 6: .15}
SIDE_P = {14: .55, 13: .25, 12: .08}
NOTRUMP_P = {14: .60, 13: .30, 12: .10, 11: .03}


class HeuristicAgent(Agent):
    """Bids the rounded sum of per-card win probabilities (crude, hand-tuned),
    then plays to hit its bid exactly: win when it still needs tricks, shed
    high cards safely once it has enough."""
    name = "heuristic"

    def estimate(self, v: BidView) -> float:
        est = 0.0
        trump_len = sum(1 for c in v.hand if c[0] == v.trump)
        for suit, rank in v.hand:
            if v.trump is None:
                est += NOTRUMP_P.get(rank, 0.0)
            elif suit == v.trump:
                est += TRUMP_P.get(rank, 0.1)
            else:
                est += SIDE_P.get(rank, 0.0)
        if v.trump is not None and trump_len > 3:
            est += 0.3 * (trump_len - 3)
        return est

    def bid(self, v: BidView) -> int:
        est = self.estimate(v)
        b = max(0, min(v.n_cards, round(est)))
        if b == v.forbidden:
            alts = [x for x in (b - 1, b + 1) if 0 <= x <= v.n_cards]
            b = min(alts, key=lambda x: abs(est - x))
        return b

    def play(self, v: PlayView):
        need = v.bids[v.seat] - v.tricks_won[v.seat]
        trump = v.trump
        strength = lambda c: c[1] + (100 if c[0] == trump else 0)
        legal = list(v.legal)

        if not v.trick:
            return max(legal, key=strength) if need > 0 else min(legal, key=strength)

        led = v.trick[0][1][0]
        best = v.trick[0][1]
        for _, c in v.trick[1:]:
            if beats(c, best, led, trump):
                best = c
        winners = [c for c in legal if beats(c, best, led, trump)]
        losers = [c for c in legal if c not in winners]
        last_to_act = len(v.trick) == v.cfg.n_players - 1

        if need > 0:
            if winners:
                return min(winners, key=strength) if last_to_act else max(winners, key=strength)
            return min(legal, key=strength)
        if losers:
            return max(losers, key=strength)
        return min(winners, key=strength)
