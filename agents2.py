"""Version A and Version B of the learned bot, as engine agents (3 players).

Version A  = learned bid model (1) + self-play-tuned heuristic (2) + tricks-needed planner (4)
Version B  = Version A + value network (3) used as the leaf evaluation of a one-trick look-ahead (card play)

All weights are loaded from models/*.npz|npy (plain numpy, no torch)."""
import os
import numpy as np
from boerenbridge import Agent
from ismcts import enc, dec, infer_voids
from tuned import Tuned
from nn import MLP
from bidnet import choose_bid
from planner import Planner
from valuenet import ValueSearch

import sys
_BASE = os.environ.get("BB_DATA") or getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(_BASE, "models") if os.path.isdir(os.path.join(_BASE, "models")) else _BASE
_cache = {}


def _load(bid="bidnet.npz", val="valuenet.npz"):
    if "T" not in _cache:
        _cache["T"] = Tuned(np.load(os.path.join(MODELS, "tuned_theta.npy")))
    for k, f in (("bid", bid), ("val", val)):
        if (k, f) not in _cache:
            _cache[(k, f)] = MLP.load(os.path.join(MODELS, f))
    return dict(T=_cache["T"], bid=_cache[("bid", bid)], val=_cache[("val", val)])


def _state(v):
    n = v.cfg.n_players
    tr = [enc(c) for _, c in v.trick]
    ldr = v.trick[0][0] if v.trick else v.seat
    return dict(n=n, seat=v.seat, trump=("CDSH".index(v.trump) if v.trump else -1),
                tc=enc(v.trump_card) if v.trump_card else None, tr=tr, ldr=ldr, bids=list(v.bids),
                won=list(v.tricks_won), tl=v.tricks_left, hand=[enc(c) for c in v.hand],
                lg=[enc(c) for c in v.legal], played={enc(c) for _, c in v.played},
                voids=infer_voids(v.played, n))


class _Base(Agent):
    name = "learned"

    def __init__(self, seed=0, bid="bidnet.npz", val="valuenet.npz"):
        m = _load(bid, val)
        self.T, self.bidnet, self.vnet = m["T"], m["bid"], m["val"]
        self.seed = seed

    def bid(self, v):
        if v.cfg.n_players != 3:
            raise ValueError("the learned bots are trained for 3 players")
        trump = "CDSH".index(v.trump) if v.trump else -1
        pos = len(v.bids_so_far)
        before = [v.bids_so_far[s] for s in sorted(v.bids_so_far, key=lambda s: _order(v, s))]
        tcr = v.trump_card[1] if v.trump_card else 0
        return choose_bid(self.bidnet, self.T, [enc(c) for c in v.hand], trump, v.n_cards, v.forbidden, before, pos, tcr)


def _order(v, s):
    """Bid order of a seat: the bids dict is filled in bidding order, so dict order is the order."""
    return list(v.bids_so_far).index(s)


class VersionA(_Base):
    name = "learned-A"

    def __init__(self, seed=0, K=16, bid="bidnet.npz", val="valuenet.npz"):
        super().__init__(seed, bid, val)
        self.planner = Planner(K=K, seed=seed)

    def play(self, v):
        return dec(self.planner.choose(_state(v)))


class VersionB(_Base):
    name = "learned-B"

    def __init__(self, seed=0, K=16, bid="bidnet.npz", val="valuenet.npz"):
        super().__init__(seed, bid, val)
        self.search = ValueSearch(self.vnet, K=K, seed=seed)

    def play(self, v):
        return dec(self.search.choose(_state(v)))


# ---- ablation steps (used in the comparison): each adds one component
class TunedAgent(_Base):
    """Item 2 only: the ES-tuned heuristic (bids and plays)."""
    name = "tuned"

    def bid(self, v):
        trump = "CDSH".index(v.trump) if v.trump else -1
        return self.T.bid([enc(c) for c in v.hand], trump, v.n_cards, v.forbidden)

    def play(self, v):
        st = _state(v)
        need = v.bids[v.seat] - v.tricks_won[v.seat]
        return dec(self.T.move(st["lg"], st["tr"], need, st["trump"], len(st["tr"]) == v.cfg.n_players - 1))


class BidNetAgent(TunedAgent):
    """Items 2 + 1: learned bid model, tuned-heuristic card play."""
    name = "tuned+bidnet"
    bid = _Base.bid
