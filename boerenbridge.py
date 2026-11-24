"""
Boerenbridge engine: 3-player, escalating-rounds variant.

Rules implemented
-----------------
- Deck: every card below `min_rank` removed (default: 6 and up -> 36 cards).
- Rounds: hand sizes go 1..max_hand with trump, then ONE round of max_hand with NO
  trump, and the game ends (max_hand=10 gives 11 rounds).
- Deal: n cards each; the next card of the undealt stack is flipped and its
  suit is trump (skipped in the no-trump round).
- Bidding: starts left of the dealer, dealer bids last. Hook rule: when the
  hand has >= hook_min_cards cards, the last bidder may not bid the number
  that makes the total of all bids equal the number of tricks.
- Play: left of the dealer leads trick 1, then the trick winner leads.
  Must follow suit. Highest trump wins, otherwise highest card of the led suit.
- Scoring per round: exact bid -> correct_base + per_trick * tricks;
  otherwise -miss_penalty * |bid - tricks|.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Optional

SUITS = "CDSH"
RANK_LABEL = {11: "J", 12: "Q", 13: "K", 14: "A"}

Card = tuple  # (suit_char, rank_int)


def card_str(c: Card) -> str:
    return f"{RANK_LABEL.get(c[1], c[1])}{c[0]}"


@dataclass(frozen=True)
class Config:
    n_players: int = 3
    min_rank: int = 6
    max_hand: int = 10
    hook_min_cards: int = 3
    correct_base: int = 10
    per_trick: int = 3
    miss_penalty: int = 3  # per trick off; ASSUMED equal to per_trick (see README)

    def deck(self) -> list[Card]:
        return [(s, r) for s in SUITS for r in range(self.min_rank, 15)]

    def validate(self) -> None:
        need = self.n_players * self.max_hand + 1  # +1 for the trump card
        if len(self.deck()) < need:
            raise ValueError(f"deck of {len(self.deck())} too small for {self.max_hand} cards each")


def round_schedule(max_hand: int) -> list[tuple[int, bool]]:
    """[(cards_per_player, trump_in_play), ...]: 1..max_hand with trump, then one no-trump round. The game ends there."""
    up = [(n, True) for n in range(1, max_hand + 1)]
    return up + [(max_hand, False)]


# ---------------------------------------------------------------- rules ---

def legal_cards(hand: list[Card], trick: list[tuple[int, Card]]) -> list[Card]:
    if not trick:
        return list(hand)
    led = trick[0][1][0]
    follow = [c for c in hand if c[0] == led]
    return follow or list(hand)


def beats(c: Card, best: Card, led: str, trump: Optional[str]) -> bool:
    if c[0] == best[0]:
        return c[1] > best[1]
    return trump is not None and c[0] == trump


def trick_winner(trick: list[tuple[int, Card]], trump: Optional[str]) -> int:
    led = trick[0][1][0]
    best_seat, best = trick[0]
    for seat, c in trick[1:]:
        if beats(c, best, led, trump):
            best_seat, best = seat, c
    return best_seat


def score_round(bid: int, won: int, cfg: Config) -> int:
    if bid == won:
        return cfg.correct_base + cfg.per_trick * won
    return -cfg.miss_penalty * abs(bid - won)


# ---------------------------------------------------------------- views ---

@dataclass
class BidView:
    seat: int
    hand: tuple
    n_cards: int
    trump: Optional[str]
    trump_card: Optional[Card]
    bids_so_far: dict          # seat -> bid, for seats that already bid
    seats_to_bid_after: int    # how many players bid after me
    forbidden: Optional[int]   # hook rule
    scores: tuple
    round_index: int
    cfg: Config


@dataclass
class PlayView:
    seat: int
    hand: tuple
    legal: tuple
    trick: tuple               # ((seat, card), ...) so far this trick
    trump: Optional[str]
    bids: tuple                # bid by seat
    tricks_won: tuple          # tricks won so far this round, by seat
    played: tuple              # every ((seat, card)) played this round
    n_cards: int
    tricks_left: int           # including the current one
    cfg: Config
    trump_card: Optional[Card] = None   # the flipped card (public; never in any hand)


class Agent:
    name = "agent"
    def bid(self, v: BidView) -> int: raise NotImplementedError
    def play(self, v: PlayView) -> Card: raise NotImplementedError
    def end_round(self, res) -> None:
        """Called after every round with the full RoundResult (all hands, bids and the play log are public by then)."""


# ---------------------------------------------------------------- rounds ---

@dataclass
class RoundResult:
    round_index: int
    n_cards: int
    trump: Optional[str]
    dealer: int
    bids: list
    won: list
    points: list
    hook_forbidden: Optional[int]   # value the last bidder was forbidden to bid
    hands: list = field(default_factory=list)
    plays: list = field(default_factory=list)   # [(seat, card), ...] in play order, every trick of the round


def play_round(rng, agents, n_cards, use_trump, dealer, cfg, scores, round_index) -> RoundResult:
    n = cfg.n_players
    deck = cfg.deck()
    rng.shuffle(deck)
    hands = [sorted(deck[i * n_cards:(i + 1) * n_cards]) for i in range(n)]
    trump_card = deck[n * n_cards] if use_trump else None
    trump = trump_card[0] if trump_card else None
    first = (dealer + 1) % n
    order = [(first + i) % n for i in range(n)]

    bids: dict[int, int] = {}
    hook_forbidden = None
    for k, seat in enumerate(order):
        forbidden = None
        if k == n - 1 and n_cards >= cfg.hook_min_cards:
            f = n_cards - sum(bids.values())
            if 0 <= f <= n_cards:
                forbidden = f
                hook_forbidden = f
        view = BidView(seat, tuple(hands[seat]), n_cards, trump, trump_card, dict(bids),
                       n - 1 - k, forbidden, tuple(scores), round_index, cfg)
        b = agents[seat].bid(view)
        if not isinstance(b, int) or not 0 <= b <= n_cards or b == forbidden:
            raise ValueError(f"illegal bid {b!r} by seat {seat} (n={n_cards}, forbidden={forbidden})")
        bids[seat] = b
    bid_by_seat = tuple(bids[s] for s in range(n))

    initial_hands = [list(h) for h in hands]
    won = [0] * n
    played: list[tuple[int, Card]] = []
    leader = first
    for t in range(n_cards):
        trick: list[tuple[int, Card]] = []
        for i in range(n):
            seat = (leader + i) % n
            legal = legal_cards(hands[seat], trick)
            view = PlayView(seat, tuple(hands[seat]), tuple(legal), tuple(trick), trump,
                            bid_by_seat, tuple(won), tuple(played), n_cards, n_cards - t, cfg, trump_card)
            c = agents[seat].play(view)
            if c not in legal:
                raise ValueError(f"illegal card {c} by seat {seat}")
            hands[seat].remove(c)
            trick.append((seat, c))
            played.append((seat, c))
        w = trick_winner(trick, trump)
        won[w] += 1
        leader = w

    points = [score_round(bid_by_seat[s], won[s], cfg) for s in range(n)]
    res = RoundResult(round_index, n_cards, trump, dealer, list(bid_by_seat), won, points,
                      hook_forbidden, initial_hands, list(played))
    for ag in agents:
        ag.end_round(res)
    return res


def play_match(rng, agents, cfg: Config, first_dealer: int = 0):
    cfg.validate()
    scores = [0] * cfg.n_players
    results = []
    for r, (n_cards, use_trump) in enumerate(round_schedule(cfg.max_hand)):
        dealer = (first_dealer + r) % cfg.n_players
        res = play_round(rng, agents, n_cards, use_trump, dealer, cfg, scores, r)
        for s in range(cfg.n_players):
            scores[s] += res.points[s]
        results.append(res)
    return scores, results
