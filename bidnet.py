"""Item 1: a learned bid-outcome model. For a hand, trump, hand size, seat position and the bids already made it predicts
the expected round points of EVERY candidate bid (hook bid masked); the agent bids the argmax.
Training data: simulated deals; the same deal is played out once per candidate bid (common random numbers), with
the bidder using the tuned heuristic's card play and the others the stock heuristic."""
import random
import numpy as np
from fast import deal, bid_round, play_out, pts, base_bidder, base_move
from tuned import Tuned, make_deals

NB = 11


def bid_features(hand, trump, nc, pos, bids_before, tc_rank, T, n=3):
    f = []
    ranks = [c & 15 for c in hand]
    if trump >= 0:
        tr = [c & 15 for c in hand if (c >> 4) == trump]
        f += [1.0 if r in tr else 0.0 for r in range(6, 15)]
        side = [c for c in hand if (c >> 4) != trump]
        sc = {}
        for c in side:
            sc[c >> 4] = sc.get(c >> 4, 0) + 1
        f += [sum(1 for c in side if (c & 15) == r) for r in (10, 11, 12, 13, 14)]
        f += [len(tr), 0.0]
        n_side_suits = [s for s in range(4) if s != trump]
        f += [sum(1 for s in n_side_suits if sc.get(s, 0) == 0), sum(1 for s in n_side_suits if sc.get(s, 0) == 1),
              max(sc.values()) if sc else 0]
        f += [tc_rank / 14.0]
    else:
        f += [0.0] * 9
        f += [sum(1 for r in ranks if r == q) for q in (10, 11, 12, 13, 14)]
        f += [0, 1.0]
        sc = {}
        for c in hand:
            sc[c >> 4] = sc.get(c >> 4, 0) + 1
        f += [sum(1 for s in range(4) if sc.get(s, 0) == 0), sum(1 for s in range(4) if sc.get(s, 0) == 1),
              max(sc.values())]
        f += [0.0]
    sb = sum(bids_before)
    est = T.est(hand, trump)
    f += [nc / 10.0, (nc / 10.0) ** 2, 1.0 if pos == 0 else 0.0, 1.0 if pos == 1 else 0.0, 1.0 if pos == n - 1 else 0.0,
          sb / nc, (nc - sb) / nc, (nc - sb) / 5.0, est / nc, est, T.raw_bid(hand, trump, nc)]
    return f


def gen_bid_data(T, deals, n=3):
    """Rows: features; Y[row, b] = hero's round points when bidding b (play by T, opponents stock heuristic)."""
    X, Y, M = [], [], []
    for nc, hands, trump, pos, tc in deals:
        bids0 = []
        for k in range(pos):
            forb = None
            bids0.append(base_bidder(hands[k], trump, nc, forb, bids0))
        feats = bid_features(hands[pos], trump, nc, pos, bids0, 0, T, n)   # trump-card rank filled by caller if wanted
        y = np.zeros(NB)
        m = np.zeros(NB)
        for b in range(nc + 1):
            bids = list(bids0) + [b]
            for k in range(pos + 1, n):
                forb = None
                if k == n - 1 and nc >= 3:
                    f = nc - sum(bids)
                    if 0 <= f <= nc:
                        forb = f
                bids.append(base_bidder(hands[k], trump, nc, forb, bids))
            if pos == n - 1 and nc >= 3 and b == nc - sum(bids0):
                continue                                    # illegal hook bid
            movers = [base_move] * n
            movers[pos] = T.move
            won = play_out(hands, trump, bids, movers, n)
            y[b] = pts(b, won[pos])
            m[b] = 1.0
        X.append(feats), Y.append(y), M.append(m)
    return np.array(X), np.array(Y), np.array(M)


def choose_bid(net, T, hand, trump, nc, forb, bids_before, pos, tc_rank, n=3):
    """Argmax of the learned expected-points vector over the legal bids."""
    x = bid_features(hand, trump, nc, pos, bids_before, tc_rank, T, n)
    q = net.predict(np.array(x))[0][:nc + 1]
    if forb is not None and 0 <= forb <= nc:
        q = q.copy()
        q[forb] = -1e9
    return int(np.argmax(q))
