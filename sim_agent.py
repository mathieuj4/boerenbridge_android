"""Round simulator that lets ONE seat use a rich agent (planner / value search) that needs the full view
(played cards, voids). Everyone else uses stock move functions. Seats = bidding positions."""
import random
from fast import DECK, pts, bid_round, base_bidder, base_move
from ismcts import trick_winner_idx


def sim_hero_round(hands, trump, tc, bids, pos, hero_move, others=base_move, n=3, on_trick_start=None):
    """hero_move(st) -> card. Returns tricks won per seat."""
    hands = [h[:] for h in hands]
    nc = len(hands[0])
    won = [0] * n
    ldr = 0
    played = set()
    voids = {s: set() for s in range(n)}
    for t in range(nc):
        tr = []
        if on_trick_start:
            on_trick_start(hands[pos][:], set(played), list(won), ldr, nc - t)
        for i in range(n):
            s = (ldr + i) % n
            h = hands[s]
            if tr:
                led = tr[0] >> 4
                lg = [x for x in h if (x >> 4) == led] or h
            else:
                lg = h
            if s == pos:
                st = dict(n=n, seat=s, trump=trump, tc=tc, tr=tr, ldr=ldr, bids=bids, won=won, tl=nc - t, hand=h[:],
                          lg=lg[:], played=played, voids=voids)
                m = hero_move(st) if len(lg) > 1 else lg[0]
            else:
                m = lg[0] if len(lg) == 1 else others(lg, tr, bids[s] - won[s], trump, len(tr) == n - 1)
            if tr and (m >> 4) != (tr[0] >> 4):
                voids[s].add(tr[0] >> 4)
            h.remove(m)
            tr.append(m)
        for m in tr:
            played.add(m)
        w = (ldr + trick_winner_idx(tr, trump)) % n
        won[w] += 1
        ldr = w
    return won
