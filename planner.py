"""Item 4: a 'tricks still needed' planner for the card play.

For every legal card it estimates the expected ROUND POINTS directly:
  * the current trick is finished in K sampled worlds (hidden hands consistent with what has been seen, the other
    players answering with the stock bid-aware rule), which gives P(I win this trick);
  * every card that stays in my hand gets an analytic win probability from counting the unseen cards that could
    beat it (hypergeometric; follow-suit and ruffing considered), and a Poisson-binomial DP turns those into a
    distribution of how many of the remaining tricks I win;
  * the card with the highest expected points (10+3b if exact, -3|diff| otherwise) is played.
No rollouts and no tree: cost per decision is K x (cards) small DPs."""
import random
from math import comb
from functools import lru_cache
from fast import DECK, pts
from ismcts import determinize, heur_move, trick_winner_idx

_C = lru_cache(maxsize=None)(lambda n, k: comb(n, k) if 0 <= k <= n else 0)


def p_unbeaten(x, unseen_cnt, r, trump, n_opp):
    """P(card x wins a trick against n_opp opponents who each hold r random unseen cards).
    unseen_cnt: dict with N, per-suit counts 'suit' (list of 4) and 'higher' function inputs (suit counts of cards above x)."""
    raise NotImplementedError


def future_probs(hand, U, r, trump, n_opp, shr=1.0):
    """Win probability of each card in `hand` (list of ints) when played as the trick's leader against n_opp
    opponents that each hold r cards drawn at random from the unseen set U."""
    N = len(U)
    if N == 0 or r == 0:
        return [1.0] * len(hand)
    bysuit = [[], [], [], []]
    for c in U:
        bysuit[c >> 4].append(c & 15)
    tcount = len(bysuit[trump]) if trump >= 0 else 0
    tot = _C(N, r)
    out = []
    for x in hand:
        s, rk = x >> 4, x & 15
        Us = len(bysuit[s])
        h = sum(1 for q in bysuit[s] if q > rk)
        A = _C(N - h, r) / tot
        if s == trump or trump < 0:
            pn = A
        else:
            B = (_C(N - Us, r) - _C(N - Us - tcount, r)) / tot
            pn = A - B
        pn = max(0.0, min(1.0, pn))
        out.append((pn ** n_opp) * shr)
    return out


def pmf_bernoulli(ps):
    d = [1.0]
    for p in ps:
        nd = [0.0] * (len(d) + 1)
        for j, q in enumerate(d):
            nd[j] += q * (1 - p)
            nd[j + 1] += q * p
        d = nd
    return d


class Planner:
    def __init__(self, K=24, shr=1.0, seed=0, fallback=None):
        self.K, self.shr = K, shr
        self.rng = random.Random(seed)
        self.fallback = fallback          # optional: card the tuned heuristic would play, wins ties

    def choose(self, st):
        lg = st["lg"]
        if len(lg) == 1:
            return lg[0]
        n, me, trump = st["n"], st["seat"], st["trump"]
        tr, ldr, bids, won = st["tr"], st["ldr"], st["bids"], st["won"]
        tl = st["tl"]                                         # tricks left including this one
        hand = st["hand"]
        opps = [s for s in range(n) if s != me]
        known = set(hand) | st["played"] | set(tr) | ({st["tc"]} if st["tc"] is not None else set())
        U = [c for c in DECK if c not in known]
        played_by = {(ldr + i) % n for i in range(len(tr))}
        counts = {s: tl - (1 if s in played_by else 0) for s in opps}
        voids = st["voids"]
        worlds = [determinize(U, counts, voids, opps, self.rng) for _ in range(self.K)]
        my_need_base = won[me]
        b = bids[me]
        n_after = n - 1 - len(tr)                              # players after me in this trick
        score = {c: 0.0 for c in lg}
        for w in worlds:
            for c in lg:
                tr2 = tr + [c]
                played_now = []
                for k in range(n_after):
                    s = (ldr + len(tr2)) % n
                    oh = w[s]
                    led = tr2[0] >> 4
                    og = [x for x in oh if (x >> 4) == led] or oh
                    m = og[0] if len(og) == 1 else heur_move(og, tr2, bids[s] - won[s], trump, len(tr2) == n - 1)
                    tr2.append(m)
                    played_now.append(m)
                win = (ldr + trick_winner_idx(tr2, trump)) % n == me
                base = my_need_base + (1 if win else 0)
                if tl == 1:
                    score[c] += pts(b, base)
                    continue
                rest = [x for x in hand if x != c]
                Uw = [x for x in U if x not in played_now]
                ps = future_probs(rest, Uw, tl - 1, trump, n - 1, self.shr)
                d = pmf_bernoulli(ps)
                score[c] += sum(q * pts(b, base + j) for j, q in enumerate(d))
        best = max(score.values())
        top = [c for c in lg if score[c] >= best - 1e-9 * self.K]
        if self.fallback in top:
            return self.fallback
        return top[0] if len(top) == 1 else max(top, key=lambda c: (score[c], -(c & 15)))
