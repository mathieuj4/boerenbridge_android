"""
ISMCTS for Boerenbridge (3 players by default; the search itself works for any number of players), written against boerenbridge.py / agents.py.

What is in here
---------------
* Card codes: card = suit_index * 16 + rank   (suits in the engine's "CDSH" order, rank 6..14).
* infer_voids / unseen_info / determinize : turn what one seat can see into sampled full deals.
* play_search  : single-observer ISMCTS for choosing a card (subset-armed UCB, availability counts).
* bid_search   : a bandit over bids; each pull samples a deal, opponents' bids, and plays the round out.
* bid_values   : the same simulation with a fixed number of samples per bid (used for the hook analysis).
* rollout policies: "random" or "bid" (bid-aware: every player steers toward their own bid).
* ExactMaxN    : exact perfect-information solver for small determinized endgames (used by pimc_vote).
* ISMCTSAgent  : plugs into the engine (bid() and play()).
"""
from __future__ import annotations
import bisect
import math
import random
from types import SimpleNamespace

from boerenbridge import SUITS, Agent, round_schedule
from agents import TRUMP_P, SIDE_P, NOTRUMP_P

SUIT_IDX = {s: i for i, s in enumerate(SUITS)}


def enc(card):
    return SUIT_IDX[card[0]] * 16 + card[1]


def dec(c):
    return (SUITS[c >> 4], c & 15)


def score_fn(cfg):
    base, per, miss = cfg.correct_base, cfg.per_trick, cfg.miss_penalty
    def f(bid, won):
        return base + per * won if bid == won else -miss * abs(bid - won)
    return f


# ------------------------------------------------------------------ rules on int cards
def trick_winner_idx(tc, trump):
    """Index (in play order) of the winning card; trump is a suit index or -1."""
    best = tc[0]
    bi = 0
    for i in range(1, len(tc)):
        c = tc[i]
        if (c >> 4) == (best >> 4):
            if c > best:
                best, bi = c, i
        elif (c >> 4) == trump:
            best, bi = c, i
    return bi


def _beats(c, best, trump):
    if (c >> 4) == (best >> 4):
        return c > best
    return (c >> 4) == trump


def heur_move(lg, tr, need, trump, last_to_act):
    """The HeuristicAgent.play rule on int cards: win while a trick is still needed, shed once it is not."""
    if not tr:
        if need > 0:
            return max(lg, key=lambda c: (c & 15) + (100 if (c >> 4) == trump else 0))
        return min(lg, key=lambda c: (c & 15) + (100 if (c >> 4) == trump else 0))
    best = tr[0]
    for c in tr[1:]:
        if _beats(c, best, trump):
            best = c
    winners = [c for c in lg if _beats(c, best, trump)]
    st = lambda c: (c & 15) + (100 if (c >> 4) == trump else 0)
    if need > 0:
        if winners:
            return min(winners, key=st) if last_to_act else max(winners, key=st)
        return min(lg, key=st)
    losers = [c for c in lg if not _beats(c, best, trump)]
    if losers:
        return max(losers, key=st)
    return min(winners, key=st)


def heur_est(hand, trump):
    """Expected tricks of a hand under the heuristic's per-card win probabilities."""
    est = 0.0
    if trump < 0:
        for c in hand:
            est += NOTRUMP_P.get(c & 15, 0.0)
        return est
    tl = 0
    for c in hand:
        r = c & 15
        if (c >> 4) == trump:
            est += TRUMP_P.get(r, 0.1)
            tl += 1
        else:
            est += SIDE_P.get(r, 0.0)
    if tl > 3:
        est += 0.3 * (tl - 3)
    return est


def heur_bid(hand, trump, nc, forbidden):
    """HeuristicAgent.bid on int cards (hand-tuned per-card win probabilities)."""
    est = heur_est(hand, trump)
    b = max(0, min(nc, round(est)))
    if b == forbidden:
        alts = [x for x in (b - 1, b + 1) if 0 <= x <= nc]
        b = min(alts, key=lambda x: abs(est - x))
    return b


def match_reward(totals, pts, me, tau):
    """Smooth proxy for 'finish the match ahead': sigmoid of my final lead over the best opponent."""
    margin = totals[me] + pts[me] - max(totals[o] + pts[o] for o in range(len(totals)) if o != me)
    return 1.0 / (1.0 + math.exp(-margin / tau))


def rollout_round(hands, tr, ldr, won, bids, trump, mode, rng, eps):
    """Play the round to the end in place. mode 'random' or 'bid' (bid-aware, with eps random moves).
    eps is a float or a per-seat list."""
    n = len(hands)
    el = eps if isinstance(eps, (list, tuple)) else (eps,) * n
    while True:
        while len(tr) < n:
            mover = (ldr + len(tr)) % n
            hand = hands[mover]
            if not hand:
                return
            if tr:
                led = tr[0] >> 4
                lg = [x for x in hand if (x >> 4) == led] or hand
            else:
                lg = hand
            if len(lg) == 1:
                m = lg[0]
            elif mode == "random" or rng.random() < el[mover]:
                m = lg[int(rng.random() * len(lg))]
            else:
                m = heur_move(lg, tr, bids[mover] - won[mover], trump, len(tr) == n - 1)
            hand.remove(m)
            tr.append(m)
        w = (ldr + trick_winner_idx(tr, trump)) % n
        won[w] += 1
        ldr = w
        tr.clear()


# ------------------------------------------------------------------ what a seat can see
def infer_voids(played, n_players=3):
    """Seats that failed to follow suit are void in the led suit (from the engine's play log)."""
    voids = {s: set() for s in range(n_players)}
    for i in range(0, len(played), n_players):
        chunk = played[i:i + n_players]
        led = chunk[0][1][0]
        for seat, c in chunk[1:]:
            if c[0] != led:
                voids[seat].add(SUIT_IDX[led])
    return voids


def unseen_info(view):
    """Cards the observer cannot place (opponents' hands + the undealt stack) and each opponent's hand size."""
    me, cfg = view.seat, view.cfg
    known = {enc(c) for c in view.hand} | {enc(c) for _, c in view.played}
    if view.trump_card is not None:
        known.add(enc(view.trump_card))
    pool = [enc(c) for c in cfg.deck() if enc(c) not in known]
    plays = [0] * cfg.n_players
    for s, _ in view.played:
        plays[s] += 1
    counts = {s: view.n_cards - plays[s] for s in range(cfg.n_players) if s != me}
    return pool, counts


def determinize(pool, counts, voids, opps, rng):
    """Sample opponents' hands consistent with hand sizes and revealed voids (any number of opponents)."""
    if not any(voids[s] for s in opps):
        sh = pool[:]
        rng.shuffle(sh)
        out, i = {}, 0
        for s in opps:
            out[s] = sh[i:i + counts[s]]
            i += counts[s]
        return out
    order = sorted(opps, key=lambda s: -len(voids[s]))             # most constrained seat first
    for _ in range(30):
        avail = pool
        out, ok = {}, True
        for s in order:
            allowed = [c for c in avail if (c >> 4) not in voids[s]]
            if len(allowed) < counts[s]:
                ok = False
                break
            pick = rng.sample(allowed, counts[s])
            ps = set(pick)
            avail = [c for c in avail if c not in ps]
            out[s] = pick
        if ok:
            return out
    sh = pool[:]                                                   # fallback: ignore voids
    rng.shuffle(sh)
    out, i = {}, 0
    for s in opps:
        out[s] = sh[i:i + counts[s]]
        i += counts[s]
    return out


# ------------------------------------------------------------------ conditioning on the bids
class WorldBank:
    """Importance-resampled pool of candidate deals. Candidates are drawn from the uniform (void-consistent)
    deal distribution and weighted by how likely each opponent's *bid* is given the hand they would hold;
    drawing from the bank approximates sampling deals consistent with the bidding."""

    def __init__(self, worlds, logw):
        mx = max(logw)
        ws = [math.exp(w - mx) for w in logw]
        self.worlds = worlds
        self.cum, t = [], 0.0
        for w in ws:
            t += w
            self.cum.append(t)
        self.total = t
        self.ess = t * t / sum(w * w for w in ws)       # effective sample size of the weighted pool

    def draw(self, rng):
        i = bisect.bisect_left(self.cum, rng.random() * self.total)
        return self.worlds[min(i, len(self.worlds) - 1)]


def make_bank(sampler, loglik, size):
    worlds, logw = [], []
    for _ in range(size):
        w = sampler()
        worlds.append(w)
        logw.append(loglik(w))
    return WorldBank(worlds, logw)


def _bid_loglik(initial_hands, known_bids, trump, sigma, om=None):
    """log P(observed bids | hands) under a soft bidder: bid ~ N(a + b*estimate, sigma).
    Default (a, b, sigma) = (0, 1, sigma) is the heuristic; with `om` they are learned per opponent."""
    ll = 0.0
    for s, h in initial_hands.items():
        if s in known_bids:
            if om is None:
                d = heur_est(h, trump) - known_bids[s]
                ll -= d * d / (2.0 * sigma * sigma)
            else:
                a, b, sg = om.bid_params(s)
                d = a + b * heur_est(h, trump) - known_bids[s]
                ll -= d * d / (2.0 * sg * sg) + math.log(sg)
    return ll


# ------------------------------------------------------------------ ISMCTS for card play
class Node:
    __slots__ = ("n", "avail", "r", "kids")

    def __init__(self, n=3):
        self.n = 0
        self.avail = 0
        self.r = [0.0] * n     # summed normalized rewards, one entry per seat
        self.kids = {}


def play_search(view, iters, rng, rollout="bid", c=0.5, eps=0.1,
                opp_model="tree", infer_bids=False, sigma=1.0, match=None, om=None):
    """Single-observer ISMCTS. Returns {card_tuple: (visits, mean_value_for_me)} over the legal cards.

    opp_model   'tree'  : opponents choose by UCB over statistics pooled across sampled worlds (max^n)
                'model' : opponents choose with the bid-aware heuristic (+ eps random), as in the rollouts
                'auto'  : like 'model' for opponents whose observed play fits the heuristic (needs `om`), else 'tree'
    infer_bids  sample opponents' hands conditioned on the bids they made (importance resampling)
    match       None, or (totals, tau): reward = smooth 'finish ahead' utility instead of round points
    om          None, or a learned OppModel: per-opponent play noise, bid model for the hand conditioning
    """
    cfg, me = view.cfg, view.seat
    n = cfg.n_players
    hand0 = [enc(x) for x in view.hand]
    trump = SUIT_IDX[view.trump] if view.trump else -1
    bids = list(view.bids)
    trick0 = [enc(cd) for _, cd in view.trick]
    leader0 = view.trick[0][0] if view.trick else me
    won0 = list(view.tricks_won)
    pool, counts = unseen_info(view)
    voids = infer_voids(view.played, n)
    opps = [s for s in range(n) if s != me]
    sc = score_fn(cfg)
    scale = float(max(cfg.correct_base + cfg.per_trick * view.n_cards, cfg.miss_penalty * view.n_cards))
    if match is not None:
        totals, tau = match
        scale = 1.0
    log = math.log
    sqrt = math.sqrt
    if opp_model == "model":
        mflag = [True] * n
    elif opp_model == "auto" and om is not None:
        mflag = [om.use_model(x) for x in range(n)]
    else:
        mflag = [False] * n
    epsl = om.eps_list(eps) if om is not None else [eps] * n

    def sampler():
        return determinize(pool, counts, voids, opps, rng)

    if infer_bids:
        played_by = {s: [enc(cd) for seat, cd in view.played if seat == s] for s in opps}
        known = {s: bids[s] for s in opps}
        def loglik(det):
            return _bid_loglik({s: det[s] + played_by[s] for s in opps}, known, trump, sigma, om)
        bank = make_bank(sampler, loglik, max(300, 2 * iters))
        draw = lambda: bank.draw(rng)
    else:
        draw = sampler

    root = Node(n)
    for _ in range(iters):
        det = draw()
        hands = [None] * n
        hands[me] = hand0[:]
        for s in opps:
            hands[s] = det[s][:]
        tr = trick0[:]
        ldr = leader0
        won = won0[:]
        node = root
        path = [root]
        expanded = False
        while True:
            mover = (ldr + len(tr)) % n
            hand = hands[mover]
            if not hand:
                break
            if tr:
                led = tr[0] >> 4
                lg = [x for x in hand if (x >> 4) == led] or hand
            else:
                lg = hand
            kids = node.kids
            if mflag[mover] and mover != me:
                if len(lg) == 1:
                    m = lg[0]
                elif rng.random() < epsl[mover]:
                    m = lg[int(rng.random() * len(lg))]
                else:
                    m = heur_move(lg, tr, bids[mover] - won[mover], trump, len(tr) == n - 1)
                child = kids.get(m)
                if child is None:
                    child = Node(n)
                    child.avail = 1
                    kids[m] = child
                    expanded = True
            else:
                unexp = [m for m in lg if m not in kids]
                for m in lg:
                    ch = kids.get(m)
                    if ch is not None:
                        ch.avail += 1
                if unexp:
                    m = unexp[int(rng.random() * len(unexp))]
                    child = Node(n)
                    child.avail = 1
                    kids[m] = child
                    expanded = True
                else:
                    best, bv = None, -1e18
                    for mm in lg:
                        ch = kids[mm]
                        v = ch.r[mover] / ch.n + c * sqrt(log(ch.avail) / ch.n)
                        if v > bv:
                            bv, best = v, mm
                    m = best
                    child = kids[m]
            hand.remove(m)
            tr.append(m)
            if len(tr) == n:
                w = (ldr + trick_winner_idx(tr, trump)) % n
                won[w] += 1
                ldr = w
                tr = []
            node = child
            path.append(child)
            if expanded:
                break
        if expanded and (tr or any(hands)):
            rollout_round(hands, tr, ldr, won, bids, trump, rollout, rng, epsl)
        pts = [sc(bids[i], won[i]) for i in range(n)]
        if match is None:
            rw = [p / scale for p in pts]
        else:
            rw = [match_reward(totals, pts, i, tau) for i in range(n)]
        for nd in path:
            nd.n += 1
            rr = nd.r
            for i in range(n):
                rr[i] += rw[i]
    out = {}
    for x in view.legal:
        ch = root.kids.get(enc(x))
        out[x] = (ch.n, ch.r[me] / ch.n * scale) if ch is not None and ch.n else (0, 0.0)
    return out


# ------------------------------------------------------------------ bidding
def _bid_context(view):
    cfg, me = view.cfg, view.seat
    n = cfg.n_players
    hand = [enc(x) for x in view.hand]
    trump = SUIT_IDX[view.trump] if view.trump else -1
    known = set(hand)
    if view.trump_card is not None:
        known.add(enc(view.trump_card))
    pool = [enc(c) for c in cfg.deck() if enc(c) not in known]
    k = n - 1 - view.seats_to_bid_after
    first = (me - k) % n
    order = [(first + i) % n for i in range(n)]
    return cfg, me, n, hand, trump, pool, first, order


def _deal_sampler(view, ctx, rng):
    cfg, me, n, hand, trump, pool, first, order = ctx
    nc = view.n_cards
    def f():
        sh = pool[:]
        rng.shuffle(sh)
        out, i = {}, 0
        for s in range(n):
            if s != me:
                out[s] = sh[i:i + nc]
                i += nc
        return out
    return f


def _sample_bids_and_hands(view, ctx, my_bid, rng, draw=None, om=None):
    cfg, me, n, hand, trump, pool, first, order = ctx
    nc = view.n_cards
    det = draw() if draw is not None else _deal_sampler(view, ctx, rng)()
    hands = [None] * n
    hands[me] = hand[:]
    for s in range(n):
        if s != me:
            hands[s] = det[s][:]
    bids = [0] * n
    for seat in order:
        if seat == me:
            bids[seat] = my_bid
        elif seat in view.bids_so_far:
            bids[seat] = view.bids_so_far[seat]
        else:
            forb = None
            if seat == order[-1] and nc >= cfg.hook_min_cards:
                f = nc - sum(bids[s] for s in order[:-1])
                if 0 <= f <= nc:
                    forb = f
            bids[seat] = (heur_bid(hands[seat], trump, nc, forb) if om is None
                          else om.bid(seat, hands[seat], trump, nc, forb))
    return hands, bids, first, trump


def _bid_reward(cfg, view, me, match):
    """Returns f(bids, won) -> reward for me, plus the reward scale."""
    sc = score_fn(cfg)
    nc = view.n_cards
    if match is None:
        scale = float(max(cfg.correct_base + cfg.per_trick * nc, cfg.miss_penalty * nc))
        return (lambda bids, won: sc(bids[me], won[me]) / scale), scale
    totals, tau = match
    return (lambda bids, won: match_reward(totals, [sc(bids[s], won[s]) for s in range(cfg.n_players)], me, tau)), 1.0


def bid_search(view, iters, rng, rollout="bid", c=0.3, eps=0.1, infer_bids=False, sigma=1.0, match=None, om=None):
    """Bandit over legal bids. Returns {bid: (visits, mean_value)}."""
    ctx = _bid_context(view)
    cfg, me, trump = ctx[0], ctx[1], ctx[4]
    nc = view.n_cards
    rew, scale = _bid_reward(cfg, view, me, match)
    arms = [b for b in range(nc + 1) if b != view.forbidden]
    cnt = {b: 0 for b in arms}
    tot = {b: 0.0 for b in arms}
    draw = None
    if infer_bids and view.bids_so_far:
        sampler = _deal_sampler(view, ctx, rng)
        known = dict(view.bids_so_far)
        bank = make_bank(sampler, lambda d: _bid_loglik(d, known, trump, sigma, om), max(400, 2 * iters))
        draw = lambda: bank.draw(rng)
    epsl = om.eps_list(eps) if om is not None else eps
    for t in range(1, iters + 1):
        unv = [b for b in arms if cnt[b] == 0]
        if unv:
            b = unv[0]
        else:
            lt = math.log(t)
            b = max(arms, key=lambda x: tot[x] / cnt[x] + c * math.sqrt(lt / cnt[x]))
        hands, bids, first, tr_ = _sample_bids_and_hands(view, ctx, b, rng, draw, om)
        won = [0] * cfg.n_players
        rollout_round(hands, [], first, won, bids, tr_, rollout, rng, epsl)
        cnt[b] += 1
        tot[b] += rew(bids, won)
    return {b: (cnt[b], tot[b] / cnt[b] * scale if cnt[b] else 0.0) for b in arms}


def _play_round_hero_search(hands, bids, first, trump, me, view, iters, rng, eps, om=None):
    """Play a determinized round out where seat `me` chooses its cards with a (light) ISMCTS and the
    others play the bid-aware heuristic. Returns tricks won."""
    cfg, nc = view.cfg, view.n_cards
    tsuit = SUITS[trump] if trump >= 0 else None
    n = cfg.n_players
    won = [0] * n
    ldr, tr, played = first, [], []
    while True:
        while len(tr) < n:
            mover = (ldr + len(tr)) % n
            hand = hands[mover]
            if not hand:
                return won
            lg = ([x for x in hand if (x >> 4) == (tr[0] >> 4)] or hand) if tr else hand
            if len(lg) == 1:
                m = lg[0]
            elif mover == me:
                pv = SimpleNamespace(
                    seat=me, cfg=cfg, hand=tuple(dec(x) for x in hand), legal=tuple(dec(x) for x in lg),
                    trick=tuple(((ldr + i) % n, dec(x)) for i, x in enumerate(tr)), trump=tsuit,
                    bids=tuple(bids), tricks_won=tuple(won), played=tuple(played), n_cards=nc,
                    trump_card=view.trump_card)
                st = play_search(pv, iters, rng, rollout="bid", eps=eps,
                                 opp_model="auto" if om is not None else "model", om=om)
                m = enc(max(st, key=lambda x: st[x][0]))
            elif rng.random() < (eps if om is None else om.eps_list(eps)[mover]):
                m = lg[int(rng.random() * len(lg))]
            else:
                m = heur_move(lg, tr, bids[mover] - won[mover], trump, len(tr) == n - 1)
            hand.remove(m)
            tr.append(m)
            played.append((mover, dec(m)))
        w = (ldr + trick_winner_idx(tr, trump)) % n
        won[w] += 1
        ldr = w
        tr = []


def refine_bids(view, cands, samples, play_iters, rng, eps=0.1, match=None, om=None):
    """Second-stage bid evaluation: for each candidate bid, play `samples` sampled rounds in which MY cards
    are chosen by a light ISMCTS rather than the heuristic. Returns {bid: mean value}."""
    ctx = _bid_context(view)
    cfg, me = ctx[0], ctx[1]
    rew, scale = _bid_reward(cfg, view, me, match)
    tot = {b: 0.0 for b in cands}
    for _ in range(samples):
        seed = rng.random()
        for b in cands:
            r2 = random.Random(seed)
            hands, bids, first, trump = _sample_bids_and_hands(view, ctx, b, r2, None, om)
            won = _play_round_hero_search(hands, bids, first, trump, me, view, play_iters, r2, eps, om)
            tot[b] += rew(bids, won)
    return {b: tot[b] / samples * scale for b in cands}


def bid_values(view, samples, rng, rollout="bid", eps=0.1):
    """Mean points for EVERY bid 0..n using the same sampled deals for all bids (forbidden bid included)."""
    ctx = _bid_context(view)
    cfg, me = ctx[0], ctx[1]
    nc = view.n_cards
    sc = score_fn(cfg)
    tot = [0.0] * (nc + 1)
    for _ in range(samples):
        seed = rng.random()
        for b in range(nc + 1):
            r2 = random.Random(seed)
            hands, bids, first, trump = _sample_bids_and_hands(view, ctx, b, r2)
            won = [0] * cfg.n_players
            rollout_round(hands, [], first, won, bids, trump, rollout, r2, eps)
            tot[b] += sc(bids[me], won[me])
    return [t / samples for t in tot]


def bid_match_values(view, samples, rng, totals, eps=0.1):
    """For every bid 0..n: (mean round points, probability of finishing the match ahead) when this is the
    LAST round and the running totals are `totals`. Ties for first count one half."""
    ctx = _bid_context(view)
    cfg, me = ctx[0], ctx[1]
    nc = view.n_cards
    sc = score_fn(cfg)
    pts_tot = [0.0] * (nc + 1)
    win_tot = [0.0] * (nc + 1)
    for _ in range(samples):
        seed = rng.random()
        for b in range(nc + 1):
            r2 = random.Random(seed)
            hands, bids, first, trump = _sample_bids_and_hands(view, ctx, b, r2)
            won = [0] * cfg.n_players
            rollout_round(hands, [], first, won, bids, trump, "bid", r2, eps)
            pts = [sc(bids[s], won[s]) for s in range(cfg.n_players)]
            fin = [totals[s] + pts[s] for s in range(cfg.n_players)]
            best_other = max(fin[o] for o in range(cfg.n_players) if o != me)
            pts_tot[b] += pts[me]
            win_tot[b] += 1.0 if fin[me] > best_other else (0.5 if fin[me] == best_other else 0.0)
    return [x / samples for x in pts_tot], [x / samples for x in win_tot]


# ------------------------------------------------------------------ exact solver for small endgames
SUITMASK = [0xFFFF << (16 * s) for s in range(4)]


class ExactMaxN:
    """Perfect-information maxn over a fully determinized position (bitmask hands, memoized)."""

    def __init__(self, bids, trump, cfg):
        self.bids, self.trump = bids, trump
        self.sc = score_fn(cfg)
        self.memo = {}

    def legal_moves(self, hand, trick, unplayed):
        if trick:
            f = hand & SUITMASK[trick[0] >> 4]
            moves = f or hand
        else:
            moves = hand
        out, last = [], -1
        for c in range(64):
            if not (moves >> c) & 1:
                continue
            if last >= 0 and (last >> 4) == (c >> 4):
                between = ((1 << c) - 1) & ~((1 << (last + 1)) - 1)
                if not (unplayed & between):
                    last = c
                    continue
            out.append(c)
            last = c
        return out

    def after(self, hands, trick, leader, won, turn, c):
        nh = list(hands)
        nh[turn] &= ~(1 << c)
        nh = tuple(nh)
        nt = trick + (c,)
        if len(nt) == 3:
            w = (leader + trick_winner_idx(nt, self.trump)) % 3
            nw = list(won)
            nw[w] += 1
            return self.search(nh, (), w, tuple(nw))
        return self.search(nh, nt, leader, won)

    def search(self, hands, trick, leader, won):
        key = (hands, trick, leader, won)
        hit = self.memo.get(key)
        if hit is not None:
            return hit
        if not trick and not any(hands):
            res = tuple(self.sc(self.bids[p], won[p]) for p in range(3))
            self.memo[key] = res
            return res
        turn = (leader + len(trick)) % 3
        unplayed = hands[0] | hands[1] | hands[2]
        for c in trick:
            unplayed |= 1 << c
        best = None
        for c in self.legal_moves(hands[turn], trick, unplayed):
            vec = self.after(hands, trick, leader, won, turn, c)
            if (best is None or vec[turn] > best[turn]
                    or (vec[turn] == best[turn] and sum(vec) - vec[turn] < sum(best) - best[turn])):
                best = vec
        self.memo[key] = best
        return best


def mask_of(cards):
    m = 0
    for c in cards:
        m |= 1 << c
    return m


def pimc_vote(view, n_samples, rng):
    """Perfect-information Monte Carlo: in each sampled deal find the best card with the exact solver
    (ties split evenly) and report each card's share of the votes."""
    cfg, me = view.cfg, view.seat
    n = cfg.n_players
    trump = SUIT_IDX[view.trump] if view.trump else -1
    bids = list(view.bids)
    trick0 = tuple(enc(cd) for _, cd in view.trick)
    leader0 = view.trick[0][0] if view.trick else me
    won0 = tuple(view.tricks_won)
    pool, counts = unseen_info(view)
    voids = infer_voids(view.played, n)
    opps = [s for s in range(n) if s != me]
    legal = [enc(x) for x in view.legal]
    votes = {c: 0.0 for c in legal}
    for _ in range(n_samples):
        det = determinize(pool, counts, voids, opps, rng)
        masks = [0, 0, 0]
        masks[me] = mask_of(enc(x) for x in view.hand)
        for s in opps:
            masks[s] = mask_of(det[s])
        solver = ExactMaxN(bids, trump, cfg)
        vals = {c: solver.after(tuple(masks), trick0, leader0, won0, me, c)[me] for c in legal}
        top = max(vals.values())
        best = [c for c in legal if vals[c] == top]
        for c in best:
            votes[c] += 1.0 / len(best)
    return {dec(c): v / n_samples for c, v in votes.items()}


# ------------------------------------------------------------------ learning the opponents during the match
class OppModel:
    """Per-opponent model fitted from what is public after every round (all hands, bids and the play log).

    bids   bid ~ N(a + b * est(hand), sigma). (a, b) is a ridge fit shrunk towards the heuristic (0, 1);
           each opponent is shrunk towards the pooled fit of all opponents, sigma from the residuals.
    play   eps = how often the opponent departs from the bid-aware heuristic card, from agreement counts
           (prior: 90% agreement worth 20 decisions). An opponent fitting the heuristic is simulated
           as heuristic + eps noise, any other as a UCB 'tree' player (opp_model='auto').
    """
    LAM_A, LAM_B, N0_SIG, SIG0, SIG_MIN = 4.0, 2.0, 4.0, 1.0, 0.5
    AGREE0, N0_AGREE, USE_EPS = 0.9, 20.0, 0.3

    def __init__(self, me=0, n=3):
        self.me, self.n = me, n
        self.obs = {s: [] for s in range(n)}          # per seat: (est, bid)
        self.agree = {s: [0, 0] for s in range(n)}    # per seat: [agreements, decisions]
        self._fit = None

    # ---- learning
    def observe(self, res):
        trump = SUIT_IDX[res.trump] if res.trump else -1
        n = self.n
        first = (res.dealer + 1) % n
        for s in range(n):
            if s == res.dealer and res.hook_forbidden is not None:
                continue                              # the last bidder's bid may have been forced by the hook rule
            self.obs[s].append((heur_est([enc(c) for c in res.hands[s]], trump), res.bids[s]))
        hands = {s: {enc(c) for c in res.hands[s]} for s in range(n)}
        won, tr, ldr = [0] * n, [], first
        for seat, card in res.plays:
            c = enc(card)
            hand = hands[seat]
            lg = [x for x in hand if tr and (x >> 4) == (tr[0] >> 4)] or sorted(hand)
            if len(lg) > 1:
                pred = heur_move(lg, tr, res.bids[seat] - won[seat], trump, len(tr) == n - 1)
                self.agree[seat][0] += int(pred == c)
                self.agree[seat][1] += 1
            hand.discard(c)
            tr.append(c)
            if len(tr) == n:
                w = (ldr + trick_winner_idx(tr, trump)) % n
                won[w] += 1
                ldr = w
                tr = []
        self._fit = None

    def _ridge(self, data, prior, lam_a, lam_b):
        sxx = sx = n = sxy = sy = 0.0
        for x, y in data:
            sxx += x * x; sx += x; sxy += x * y; sy += y; n += 1
        a0, b0 = prior
        m00, m01, m11 = n + lam_a, sx, sxx + lam_b
        r0, r1 = sy + lam_a * a0, sxy + lam_b * b0
        det = m00 * m11 - m01 * m01
        return (m11 * r0 - m01 * r1) / det, (m00 * r1 - m01 * r0) / det

    def _compute(self):
        pooled = [d for s in range(self.n) if s != self.me for d in self.obs[s]]
        pa, pb = self._ridge(pooled, (0.0, 1.0), self.LAM_A, self.LAM_B)
        fit = {}
        for s in range(self.n):
            if s == self.me:
                continue
            d = self.obs[s]
            a, b = self._ridge(d, (pa, pb), 2 * self.LAM_A, 2 * self.LAM_B)
            sse = sum((y - a - b * x) ** 2 for x, y in d)
            sg = max(self.SIG_MIN, math.sqrt((sse + self.N0_SIG * self.SIG0 ** 2) / (len(d) + self.N0_SIG)))
            ag, nd = self.agree[s]
            eps = max(0.1, 1.0 - (ag + self.AGREE0 * self.N0_AGREE) / (nd + self.N0_AGREE))
            fit[s] = (a, b, sg, eps)
        self._fit = fit

    def params(self, s):
        if self._fit is None:
            self._compute()
        return self._fit[s]

    # ---- use
    def bid_params(self, s):
        a, b, sg, _ = self.params(s)
        return a, b, sg

    def eps_list(self, base):
        out = []
        for s in range(self.n):
            out.append(base if s == self.me else self.params(s)[3])
        return out

    def use_model(self, s):
        return s != self.me and self.params(s)[3] <= self.USE_EPS

    def bid(self, s, hand, trump, nc, forbidden):
        a, b, _ = self.bid_params(s)
        est = a + b * heur_est(hand, trump)
        x = max(0, min(nc, round(est)))
        if x == forbidden:
            alts = [y for y in (x - 1, x + 1) if 0 <= y <= nc]
            x = min(alts, key=lambda y: abs(est - y))
        return x

    def summary(self):
        return {s: tuple(round(v, 2) for v in self.params(s)) for s in range(self.n) if s != self.me}


# ------------------------------------------------------------------ engine agent
class ISMCTSAgent(Agent):
    """Engine agent.

    search_bid / search_play   switch each half off (falling back to the heuristic)
    infer_bids                 condition sampled opponent hands on the bids they made
    opp_model                  'tree' (pooled UCB for opponents) or 'model' (bid-aware heuristic opponents)
    refine                     number of top bids (>= 2) re-evaluated with my own card play searched; 0 = off
    adapt                      learn each opponent's bidding and play noise during the match (OppModel) and use
                               it for hand conditioning, simulated bids and opponent move models (opp_model 'auto')
    match_aware                in the last `match_last` rounds optimize the chance of finishing ahead
                               (sigmoid of the final lead, tau = match_tau in the last round, match_tau_prev before it)
    """

    def __init__(self, play_iters=100, bid_iters=400, rollout="bid", seed=0,
                 c_play=0.5, c_bid=0.3, eps=0.1, search_bid=True, search_play=True,
                 infer_bids=False, sigma=1.0, opp_model="tree", refine=0, refine_samples=40,
                 refine_iters=8, match_aware=False, match_last=2, match_tau=1.0, match_tau_prev=6.0, adapt=False, name=None):
        from agents import HeuristicAgent
        self.play_iters, self.bid_iters, self.rollout = play_iters, bid_iters, rollout
        self.c_play, self.c_bid, self.eps = c_play, c_bid, eps
        self.search_bid, self.search_play = search_bid, search_play
        self.infer_bids, self.sigma, self.opp_model = infer_bids, sigma, opp_model
        self.refine, self.refine_samples, self.refine_iters = refine, refine_samples, refine_iters
        self.match_aware, self.match_last = match_aware, match_last
        self.match_tau, self.match_tau_prev = match_tau, match_tau_prev   # sharp in the last round, softer one round earlier
        self.adapt, self.om = adapt, None
        if adapt:
            self.opp_model = "auto"
        self.rng = random.Random(seed)
        self._h = HeuristicAgent()
        self._totals, self._ridx = (0, 0, 0), 0
        self.name = name or f"ismcts-{rollout}"

    def _match(self, cfg):
        if not self.match_aware:
            return None
        left = len(round_schedule(cfg.max_hand)) - 1 - self._ridx
        if left >= self.match_last:
            return None
        return (self._totals, self.match_tau if left == 0 else self.match_tau_prev)

    def bid(self, v):
        self._totals, self._ridx = tuple(v.scores), v.round_index
        if self.adapt and (self.om is None or v.round_index == 0):
            self.om = OppModel(v.seat, v.cfg.n_players)
        if not self.search_bid:
            return self._h.bid(v)
        m = self._match(v.cfg)
        stats = bid_search(v, self.bid_iters, self.rng, rollout="bid", c=self.c_bid, eps=self.eps,
                           infer_bids=self.infer_bids, sigma=self.sigma, match=m, om=self.om)
        choice = max(stats, key=lambda b: stats[b][0])
        if self.refine >= 2 and len(stats) >= 2:
            cands = sorted(stats, key=lambda b: -stats[b][0])[:self.refine]
            vals = refine_bids(v, cands, self.refine_samples, self.refine_iters, self.rng, self.eps, match=m, om=self.om)
            choice = max(cands, key=lambda b: (vals[b], stats[b][0]))
        return choice

    def play(self, v):
        if len(v.legal) == 1:
            return v.legal[0]
        if not self.search_play:
            return self._h.play(v)
        stats = play_search(v, self.play_iters, self.rng, rollout=self.rollout, c=self.c_play, eps=self.eps,
                            opp_model=self.opp_model, infer_bids=self.infer_bids, sigma=self.sigma,
                            match=self._match(v.cfg), om=self.om)
        return max(stats, key=lambda x: stats[x][0])

    def end_round(self, res):
        if self.om is not None:
            self.om.observe(res)
