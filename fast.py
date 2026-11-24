"""Fast integer-card round simulator shared by the training code of the learned bots.

Conventions inside the simulator: seats are numbered by BIDDING POSITION (0 = first bidder = also leads the first
trick; n-1 = dealer = last bidder, the one the hook rule binds). A policy is a function
    move(lg, trick, need, trump, last_to_act) -> card      (lg = legal cards, ints)
"""
import random
from ismcts import trick_winner_idx, heur_move, heur_bid

DECK = [s * 16 + r for s in range(4) for r in range(6, 15)]


def deal(rng, nc, use_trump, n=3):
    d = DECK[:]
    rng.shuffle(d)
    hands = [d[i * nc:(i + 1) * nc] for i in range(n)]
    tc = d[n * nc] if use_trump else None
    return hands, tc, (tc >> 4 if tc is not None else -1)


def pts(bid, won):
    return 10 + 3 * won if bid == won else -3 * abs(bid - won)


def bid_round(hands, trump, nc, bidders, n=3, hook_min=3):
    """bidders[pos](hand, trump, nc, forbidden, bids_so_far) -> bid ; returns list of bids by position."""
    bids = []
    for k in range(n):
        forb = None
        if k == n - 1 and nc >= hook_min:
            f = nc - sum(bids)
            if 0 <= f <= nc:
                forb = f
        bids.append(bidders[k](hands[k], trump, nc, forb, bids))
    return bids


def play_out(hands, trump, bids, movers, n=3, ldr=0, won=None, tr=None):
    """Play a round to the end (hands are copied). Returns tricks won per seat."""
    hands = [h[:] for h in hands]
    won = list(won) if won else [0] * n
    tr = list(tr) if tr else []
    nc = max(len(h) for h in hands)
    while True:
        while len(tr) < n:
            mover = (ldr + len(tr)) % n
            hand = hands[mover]
            if not hand:
                return won
            if tr:
                led = tr[0] >> 4
                lg = [x for x in hand if (x >> 4) == led] or hand
            else:
                lg = hand
            m = lg[0] if len(lg) == 1 else movers[mover](lg, tr, bids[mover] - won[mover], trump, len(tr) == n - 1)
            hand.remove(m)
            tr.append(m)
        w = (ldr + trick_winner_idx(tr, trump)) % n
        won[w] += 1
        ldr = w
        tr = []


def base_bidder(hand, trump, nc, forb, bids):
    return heur_bid(hand, trump, nc, forb)


def base_move(lg, tr, need, trump, last):
    return heur_move(lg, tr, need, trump, last)
