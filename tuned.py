"""Item 2: the hand-written heuristic, re-parametrised and tuned by self-play (evolution strategy).

theta0 reproduces HeuristicAgent exactly (checked in test_new.py). Parameters: the per-card win-probability tables
(trump 6..A, side J..A, no-trump 10..A), the long-trump bonus, a bid scale/offset (+ a hand-size slope) and
three play parameters (how reluctant to lead trump, and when to win cheaply instead of with the highest card)."""
import math
import random
import numpy as np
from ismcts import _beats
from agents import TRUMP_P, SIDE_P, NOTRUMP_P
from fast import deal, pts, bid_round, play_out, base_bidder, base_move

TR_R = list(range(6, 15))
SD_R = [11, 12, 13, 14]
NT_R = [10, 11, 12, 13, 14]
NAMES = ([f"trump_{r}" for r in TR_R] + [f"side_{r}" for r in SD_R] + [f"nt_{r}" for r in NT_R]
         + ["long_trump_bonus", "bid_scale", "bid_offset", "bid_slope", "lead_trump_discount", "cheap_win_g"])


def theta0():
    return np.array([TRUMP_P[r] for r in TR_R] + [SIDE_P.get(r, 0.0) for r in SD_R] + [NOTRUMP_P.get(r, 0.0) for r in NT_R]
                    + [0.3, 1.0, 0.0, 0.0, 0.0, 0.0])


class Tuned:
    """Parametric heuristic on int cards."""

    def __init__(self, th):
        th = np.asarray(th, float)
        self.th = th
        self.tp = {r: th[i] for i, r in enumerate(TR_R)}
        self.sp = {r: th[9 + i] for i, r in enumerate(SD_R)}
        self.np_ = {r: th[13 + i] for i, r in enumerate(NT_R)}
        self.ltb, self.scale, self.off, self.slope, self.lead_d, self.g = [float(x) for x in th[18:24]]
        self.thr = 1.0001 - max(0.0, self.g)

    def pc(self, c, trump):
        """The table's win probability for one card."""
        r = c & 15
        if trump < 0:
            return self.np_.get(r, 0.0)
        if (c >> 4) == trump:
            return self.tp[r]
        return self.sp.get(r, 0.0)

    def est(self, hand, trump):
        if trump < 0:
            return sum(self.np_.get(c & 15, 0.0) for c in hand)
        e, tl = 0.0, 0
        for c in hand:
            if (c >> 4) == trump:
                e += self.tp[c & 15]
                tl += 1
            else:
                e += self.sp.get(c & 15, 0.0)
        if tl > 3:
            e += self.ltb * (tl - 3)
        return e

    def raw_bid(self, hand, trump, nc):
        return self.scale * self.est(hand, trump) + self.off + self.slope * (nc - 5) / 5.0

    def bid(self, hand, trump, nc, forb, bids=None):
        x = self.raw_bid(hand, trump, nc)
        b = max(0, min(nc, round(x)))
        if b == forb:
            alts = [a for a in (b - 1, b + 1) if 0 <= a <= nc]
            b = min(alts, key=lambda a: abs(x - a))
        return b

    def move(self, lg, tr, need, trump, last):
        ld = self.lead_d
        st = lambda c: (c & 15) + (100 if (c >> 4) == trump else 0)
        if not tr:
            if need > 0:
                return max(lg, key=lambda c: (c & 15) + (100 * (1 - ld) if (c >> 4) == trump else 0))
            return min(lg, key=st)
        best = tr[0]
        for c in tr[1:]:
            if _beats(c, best, trump):
                best = c
        winners = [c for c in lg if _beats(c, best, trump)]
        if need > 0:
            if winners:
                if last:
                    return min(winners, key=st)
                cheap = [c for c in winners if self.pc(c, trump) >= self.thr]
                return min(cheap, key=st) if cheap else max(winners, key=st)
            return min(lg, key=st)
        losers = [c for c in lg if c not in winners]
        return max(losers, key=st) if losers else min(winners, key=st)


# ------------------------------------------------------------------ self-play objective
def make_deals(seed, count, n=3, max_hand=10):
    rng = random.Random(seed)
    sched = [(k, True) for k in range(1, max_hand + 1)] + [(max_hand, False)]
    out = []
    for _ in range(count):
        nc, ut = rng.choice(sched)
        hands, tc, trump = deal(rng, nc, ut, n)
        out.append((nc, hands, trump, rng.randrange(n), tc))
    return out


def eval_theta(th, deals, n=3, opp_bid=base_bidder, opp_move=base_move):
    """Mean round points of the tuned player (random seat) against heuristic opponents, on fixed deals."""
    T = Tuned(th)
    tot = 0.0
    for nc, hands, trump, pos, *_ in deals:
        bidders = [opp_bid] * n
        bidders[pos] = T.bid
        bids = bid_round(hands, trump, nc, bidders, n)
        movers = [opp_move] * n
        movers[pos] = T.move
        won = play_out(hands, trump, bids, movers, n)
        tot += pts(bids[pos], won[pos])
    return tot / len(deals)


def _ev(args):
    return eval_theta(*args)


def es_tune(theta_init, iters=40, pop=16, sigma=None, deals_per_iter=500, seed=0, procs=2, lr=0.5, log=print):
    """Antithetic evolution strategy with fresh common-random-number deals every iteration."""
    from multiprocessing import get_context
    rng = np.random.default_rng(seed)
    th = theta_init.copy()
    if sigma is None:
        sigma = np.array([0.05] * 18 + [0.1, 0.08, 0.2, 0.2, 0.1, 0.1])
    m, v = np.zeros_like(th), np.zeros_like(th)
    hist = []
    best, best_val = th.copy(), -1e9
    with get_context("fork").Pool(procs) as pool:
        for it in range(iters):
            deals = make_deals(10_000_000 + seed * 1000 + it, deals_per_iter)
            eps = rng.standard_normal((pop, len(th)))
            cands = [th] + [th + sigma * e for e in eps] + [th - sigma * e for e in eps]
            vals = np.array(pool.map(_ev, [(c, deals) for c in cands]))
            base, plus, minus = vals[0], vals[1:1 + pop], vals[1 + pop:]
            grad = ((plus - minus)[:, None] * eps).mean(0) / (2 * 1.0) / sigma    # d value / d theta (per sigma unit)
            g = grad * sigma ** 2 / np.maximum(sigma, 1e-9)                          # natural scaling
            m = 0.8 * m + 0.2 * g
            th = th + lr * m / (np.abs(m).max() + 1e-9) * sigma * 1.0
            th[18] = max(th[18], 0.0)
            th[22] = max(th[22], 0.0)
            hist.append((it, float(base), float(np.max(vals))))
            if it % 5 == 0 or it == iters - 1:
                log(f"it {it:3d} base {base:6.2f} best-in-pop {np.max(vals):6.2f}")
    return th, hist
