"""Item 3: a value network V(state) = expected FINAL round points of the bidder from a position between two tricks,
if it keeps playing the planner. It replaces rollouts inside a one-trick look-ahead: for every candidate card,
K sampled worlds finish the current trick, V scores the resulting state, and the best average wins."""
import random
import numpy as np
from fast import DECK, pts, bid_round, base_bidder, base_move
from planner import future_probs, pmf_bernoulli, Planner
from sim_agent import sim_hero_round
from bidnet import choose_bid
from tuned import Tuned, make_deals
from ismcts import determinize, heur_move, trick_winner_idx


def leaf_features(hand, U, tl, won, bids, ldr, me, trump, tcr, n=3):
    """State after a completed trick (or at the deal): hand has tl cards, tl tricks remain."""
    b, w = bids[me], won[me]
    ps = future_probs(hand, U, tl, trump, n - 1)
    d = pmf_bernoulli(ps)
    d = d + [0.0] * (11 - len(d))
    ep = sum(q * pts(b, w + j) for j, q in enumerate(d))
    need = b - w
    rel = (ldr - me) % n
    onee = [(bids[s] - won[s]) for s in range(n) if s != me]
    nt_hand = sum(1 for c in hand if (c >> 4) == trump) if trump >= 0 else 0
    nt_un = sum(1 for c in U if (c >> 4) == trump) if trump >= 0 else 0
    return (d[:11] + sorted(ps, reverse=True)[:10] + [0.0] * (10 - len(ps))
            + [ep, sum(ps), need, need / tl, 1.0 if need > tl else 0.0, 1.0 if need < 0 else 0.0, tl / 10.0,
               1.0 if rel == 0 else 0.0, 1.0 if rel == 1 else 0.0, 1.0 if rel == 2 else 0.0]
            + [max(-3, min(6, x)) / tl for x in onee] + [nt_hand, nt_un, tcr / 14.0, 1.0 if trump < 0 else 0.0])


def gen_value_data(args):
    """Play rounds with the full Version-A policy (learned bid + planner) and record every between-tricks state."""
    seed, count, theta_path, bid_path = args
    from nn import MLP
    T = Tuned(np.load(theta_path))
    net = MLP.load(bid_path)
    deals = make_deals(seed, count)
    P = Planner(K=12, seed=seed)
    X, Y = [], []
    for nc, hands, trump, pos, tc in deals:
        tcr = (tc & 15) if tc is not None else 0
        bids = []
        for k in range(3):
            forb = None
            if k == 2 and nc >= 3:
                f = nc - sum(bids)
                if 0 <= f <= nc:
                    forb = f
            if k == pos:
                bids.append(choose_bid(net, T, hands[k], trump, nc, forb, list(bids), pos, tcr))
            else:
                bids.append(base_bidder(hands[k], trump, nc, forb, bids))
        rows = []

        def rec(hand, played, won, ldr, tl):
            known = set(hand) | played | ({tc} if tc is not None else set())
            U = [c for c in DECK if c not in known]
            rows.append(leaf_features(hand, U, tl, won, bids, ldr, pos, trump, tcr))

        won = sim_hero_round(hands, trump, tc, bids, pos, lambda st: P.choose(st), on_trick_start=rec)
        y = pts(bids[pos], won[pos])
        X += rows
        Y += [y] * len(rows)
    return np.array(X), np.array(Y)


class ValueSearch:
    """One-trick look-ahead with the value network at the leaves."""

    def __init__(self, vnet, K=16, seed=0, fallback_planner=None):
        self.v, self.K = vnet, K
        self.rng = random.Random(seed)

    def choose(self, st):
        lg = st["lg"]
        if len(lg) == 1:
            return lg[0]
        n, me, trump = st["n"], st["seat"], st["trump"]
        tr, ldr, bids, won, tl, hand = st["tr"], st["ldr"], st["bids"], st["won"], st["tl"], st["hand"]
        tc = st["tc"]
        tcr = (tc & 15) if tc is not None else 0
        opps = [s for s in range(n) if s != me]
        known = set(hand) | st["played"] | set(tr) | ({tc} if tc is not None else set())
        U = [c for c in DECK if c not in known]
        played_by = {(ldr + i) % n for i in range(len(tr))}
        counts = {s: tl - (1 if s in played_by else 0) for s in opps}
        worlds = [determinize(U, counts, st["voids"], opps, self.rng) for _ in range(self.K)]
        n_after = n - 1 - len(tr)
        b = bids[me]
        feats, owner, exact = [], [], {c: 0.0 for c in lg}
        for w in worlds:
            for c in lg:
                tr2 = tr + [c]
                pn = []
                for k in range(n_after):
                    s = (ldr + len(tr2)) % n
                    oh = w[s]
                    led = tr2[0] >> 4
                    og = [x for x in oh if (x >> 4) == led] or oh
                    m = og[0] if len(og) == 1 else heur_move(og, tr2, bids[s] - won[s], trump, len(tr2) == n - 1)
                    tr2.append(m)
                    pn.append(m)
                wi = (ldr + trick_winner_idx(tr2, trump)) % n
                won2 = list(won)
                won2[wi] += 1
                if tl == 1:
                    exact[c] += pts(b, won2[me])
                    continue
                rest = [x for x in hand if x != c]
                Uw = [x for x in U if x not in pn]
                feats.append(leaf_features(rest, Uw, tl - 1, won2, bids, wi, me, trump, tcr, n))
                owner.append(c)
        if tl == 1:
            return max(lg, key=lambda c: (exact[c], -(c & 15)))
        pred = self.v.predict(np.array(feats))[:, 0]
        score = {c: 0.0 for c in lg}
        for c, p in zip(owner, pred):
            score[c] += p
        return max(lg, key=lambda c: (score[c], -(c & 15)))
