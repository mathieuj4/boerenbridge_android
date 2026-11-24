"""Play Boerenbridge against the bots in your browser.

    python3 play.py            # then open http://localhost:8000  (opens automatically)
    python3 play.py --port 8123 --no-browser

Standard library only (plus the project files next to this one). The engine runs in a worker thread;
the browser polls /api/state and answers prompts through /api/bid, /api/play and /api/next.
"""
import argparse, json, os, random, sys, threading, time, traceback, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("BB_DATA") or getattr(sys, "_MEIPASS", HERE)          # where ui/ lives (inside the bundle when run as the .exe)
sys.path.insert(0, HERE)
from boerenbridge import Agent, Config, play_match, round_schedule, trick_winner, SUITS
from agents import HeuristicAgent, RandomAgent
from opponents import BoldHeuristic, NoisyHeuristic
from ismcts import ISMCTSAgent
from agents2 import VersionA, VersionB

CFGS = {3: Config(), 4: Config(n_players=4, max_hand=8)}   # 4 players: at most 8 cards (8 rounds with trump + 1 without)

# key -> (label, description, factory(seed))
BOTS = {
    "random":    ("Random", "Plays any legal card, bids anything legal.", lambda s: RandomAgent(random.Random(s))),
    "heuristic": ("Heuristic", "Hand-tuned rules: bids the rounded sum of card strengths, then plays to hit its bid. Underbids a lot.",
                  lambda s: HeuristicAgent()),
    "bold":      ("Bold heuristic", "The heuristic, but bids one trick too many.", lambda s: BoldHeuristic(1.0)),
    "noisy":     ("Noisy heuristic", "The heuristic, but plays a random legal card 35% of the time.", lambda s: NoisyHeuristic(0.35, s)),
    "ismcts-fast": ("ISMCTS (fast)", "Search bot, 100 simulations per card, 400 per bid. Near-instant.",
                    lambda s: ISMCTSAgent(play_iters=100, bid_iters=400, seed=s)),
    "ismcts":    ("ISMCTS", "Search bot as in the notebook: 300 simulations per card, 1000 per bid.",
                  lambda s: ISMCTSAgent(play_iters=300, bid_iters=1000, seed=s)),
    "ismcts-adaptive": ("ISMCTS adaptive", "ISMCTS that learns how YOU bid and play during the game and conditions on your bids (section 6.6). The strongest version.",
                        lambda s: ISMCTSAgent(play_iters=300, bid_iters=1000, seed=s, infer_bids=True, adapt=True)),
    "learned-a": ("Learned A", "Learned bid model + self-play-tuned heuristic + 'tricks still needed' planner. Very fast; 3-player games only.",
                  lambda s: VersionA(seed=s)),
    "learned-b": ("Learned B", "Version A plus a value network that scores the position after each candidate card. Very fast; 3-player games only.",
                  lambda s: VersionB(seed=s)),
    "ismcts-max": ("ISMCTS adaptive (max)", "Adaptive bot with 1000 simulations per card and 3000 per bid. Slower (a second or two per move).",
                   lambda s: ISMCTSAgent(play_iters=1000, bid_iters=3000, seed=s, infer_bids=True, adapt=True)),
}

THREE_ONLY = {"learned-a", "learned-b"}


class GameAborted(Exception):
    pass


class Game:
    def __init__(self, opp_keys, me, pace, seed, n=3):
        self.n, self.cfg = n, CFGS[n]
        self.cv = threading.Condition()
        self.me, self.pace, self.aborted = me, pace, False
        self.inbox = None
        self.rng = random.Random(seed)
        labels = [BOTS[k][0] for k in opp_keys]
        names, opp_i = [], 0
        for s in range(n):
            if s == me:
                names.append("You")
            else:
                lab = labels[opp_i]
                if labels.count(labels[opp_i]) > 1:
                    lab += " " + "ABC"[opp_i]
                names.append(lab)
                opp_i += 1
        self.names = names
        self.s = dict(status="running", me=me, n=n, names=names, n_rounds=len(round_schedule(self.cfg.max_hand)), leader=None,
                      round_index=-1, n_cards=0, trump=None, trump_card=None, dealer=None, hand=[], bids={},
                      trick=[], trick_winner=None, tricks_won=[0] * n, scores=[0] * n, prompt=None,
                      thinking=None, round_result=None, history=[], error=None, turn=None, final=None, seq=0)
        agents, k = [], 0
        for s in range(n):
            if s == me:
                inner = Human(self)
            else:
                inner = BOTS[opp_keys[k]][2](seed * 7 + s)
                k += 1
            agents.append(Seat(self, inner))
        self.agents = agents
        self.first_dealer = self.rng.randrange(n)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    # ---- worker
    def _run(self):
        try:
            scores, results = play_match(self.rng, self.agents, self.cfg, first_dealer=self.first_dealer)
            with self.cv:
                best = max(scores)
                self.s.update(status="finished", scores=list(scores), prompt=None, thinking=None, turn=None,
                              final=dict(scores=list(scores), winners=[i for i, x in enumerate(scores) if x == best]))
                self.s["seq"] += 1
        except GameAborted:
            pass
        except Exception:
            with self.cv:
                self.s.update(status="error", error=traceback.format_exc(), prompt=None, thinking=None)

    def abort(self):
        with self.cv:
            self.aborted = True
            self.cv.notify_all()

    def sleep(self, secs):
        end = time.time() + secs
        while time.time() < end:
            if self.aborted:
                raise GameAborted()
            time.sleep(0.03)

    def ask(self, prompt):
        """Block the worker until the browser answers `prompt`."""
        with self.cv:
            self.inbox = None
            self.s["prompt"] = prompt
            self.s["turn"] = self.me
            self.s["seq"] += 1
            while self.inbox is None:
                if self.aborted:
                    raise GameAborted()
                self.cv.wait(0.2)
            ans, self.inbox = self.inbox, None
            self.s["prompt"] = None
            self.s["turn"] = None
            self.s["seq"] += 1
            return ans

    def answer(self, kind, value):
        with self.cv:
            p = self.s["prompt"]
            if not p or p["type"] != kind:
                return "not your turn"
            if kind == "bid":
                if not isinstance(value, int) or not 0 <= value <= p["n"] or value == p["forbidden"]:
                    return "illegal bid"
            elif kind == "play":
                value = (value[0], int(value[1]))
                if value not in [tuple(c) for c in p["legal"]]:
                    return "illegal card"
            self.inbox = value
            self.cv.notify_all()
            return None

    # ---- state updates (called from the worker)
    def begin_round(self, v):
        with self.cv:
            s = self.s
            if v.round_index != s["round_index"]:
                s.update(round_index=v.round_index, n_cards=v.n_cards, trump=v.trump, trump_card=v.trump_card,
                         bids={}, trick=[], trick_winner=None, tricks_won=[0] * self.n, round_result=None,
                         hand=[], scores=list(v.scores), dealer=None, leader=None)
                s["_first_seen"] = False
            if not s["_first_seen"]:
                # the first bidder sits left of the dealer; the first bid call of a round tells us who that is
                if not v.bids_so_far:
                    s["dealer"] = (v.seat - 1) % self.n
                    s["leader"] = v.seat                      # left of the dealer leads the first trick
                s["_first_seen"] = True
            s["seq"] += 1

    def set_thinking(self, seat):
        with self.cv:
            self.s["thinking"] = seat
            self.s["turn"] = seat
            self.s["seq"] += 1

    def add_bid(self, seat, b):
        with self.cv:
            self.s["bids"][seat] = b
            self.s["thinking"] = None
            self.s["turn"] = None
            self.s["seq"] += 1

    def add_play(self, seat, card, v):
        with self.cv:
            s = self.s
            s["thinking"] = None
            s["turn"] = None
            if seat == self.me:
                s["hand"] = [c for c in s["hand"] if tuple(c) != tuple(card)]
            s["trick"] = [(sd, c) for sd, c in v.trick] + [(seat, card)]
            s["seq"] += 1
            done = len(s["trick"]) == self.n
            if done:
                w = trick_winner([(sd, tuple(c)) for sd, c in s["trick"]], s["trump"])
                s["tricks_won"][w] += 1
                s["trick_winner"] = w
                s["seq"] += 1
        if done:
            self.sleep(max(0.4, self.pace * 2.2))
            with self.cv:
                s["trick"], s["trick_winner"], s["leader"] = [], None, w
                s["seq"] += 1

    def finish_round(self, res):
        with self.cv:
            s = self.s
            if s["round_result"] is not None and s["round_result"]["round"] == res.round_index:
                return
            s["scores"] = [a + b for a, b in zip(s["scores"], res.points)]
            rr = dict(round=res.round_index, n_cards=res.n_cards, trump=res.trump, bids=list(res.bids),
                      won=list(res.won), points=list(res.points), hands=[[list(c) for c in h] for h in res.hands],
                      hook_forbidden=res.hook_forbidden, last=res.round_index == s["n_rounds"] - 1)
            s["round_result"] = rr
            s["history"].append(dict(round=res.round_index, n_cards=res.n_cards, trump=res.trump, bids=rr["bids"],
                                     won=rr["won"], points=rr["points"]))
            s["seq"] += 1

    def snapshot(self):
        with self.cv:
            return json.loads(json.dumps({k: v for k, v in self.s.items() if not k.startswith("_")}))


class Human:
    def __init__(self, game):
        self.g = game

    def bid(self, v):
        self.g.s["hand"] = [list(c) for c in v.hand]
        return self.g.ask(dict(type="bid", n=v.n_cards, forbidden=v.forbidden))

    def play(self, v):
        self.g.s["hand"] = [list(c) for c in v.hand]
        c = self.g.ask(dict(type="play", legal=[list(x) for x in v.legal]))
        return tuple(c)

    def end_round(self, res):
        pass


class Seat(Agent):
    """Every seat (the human's too) goes through this wrapper, which publishes what the table shows."""

    def __init__(self, game, inner):
        self.g, self.inner = game, inner

    def bid(self, v):
        g = self.g
        g.begin_round(v)
        if v.seat != g.me:
            g.set_thinking(v.seat)
        b = self.inner.bid(v)
        g.add_bid(v.seat, b)
        if v.seat != g.me:
            g.sleep(g.pace)
        return b

    def play(self, v):
        g = self.g
        with g.cv:
            g.s["tricks_won"] = list(v.tricks_won)
        if v.seat != g.me:
            g.set_thinking(v.seat)
        c = self.inner.play(v)
        g.add_play(v.seat, c, v)
        if v.seat != g.me and len(v.trick) != g.n - 1:
            g.sleep(g.pace)
        return c

    def end_round(self, res):
        g = self.g
        g.finish_round(res)
        if hasattr(self.inner, "end_round"):
            self.inner.end_round(res)
        if self.inner.__class__ is Human:
            g.ask(dict(type="next", last=res.round_index == g.s["n_rounds"] - 1))


# ------------------------------------------------------------------ http
GAME = None
LOCK = threading.Lock()
PACES = {"fast": 0.15, "normal": 0.6, "slow": 1.2}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            page_path = os.path.join(BASE, "ui", "index.html")
            if not os.path.exists(page_path):
                page_path = os.path.join(BASE, "index.html")
            with open(page_path, "r", encoding="utf-8") as f:
                page = f.read()
            bots = [dict(key=k, label=v[0], desc=v[1]) for k, v in BOTS.items()]
            page = page.replace("/*BOTS*/[]", json.dumps(bots))     # the page needs no second request to list the bots
            return self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
        if self.path == "/api/bots":
            return self._send(200, [dict(key=k, label=v[0], desc=v[1]) for k, v in BOTS.items()])
        if self.path == "/api/state":
            return self._send(200, GAME.snapshot() if GAME else {"status": "idle"})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        global GAME
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/api/new":
            opp = body.get("opponents", ["heuristic", "heuristic"])
            n = body.get("players", 3)
            if n not in CFGS:
                return self._send(400, {"error": "players must be 3 or 4"})
            if len(opp) != n - 1 or any(k not in BOTS for k in opp):
                return self._send(400, {"error": "bad opponents"})
            if n != 3 and any(k in THREE_ONLY for k in opp):
                return self._send(400, {"error": "Learned A/B are trained for 3 players only - pick other opponents for a 4-player game."})
            me = body.get("seat", 0)
            seed = body.get("seed")
            seed = int(seed) if str(seed or "").strip().lstrip("-").isdigit() else random.randrange(10 ** 6)
            if me not in range(n):
                me = random.Random(seed).randrange(n)
            with LOCK:
                if GAME:
                    GAME.abort()
                GAME = Game(opp, me, PACES.get(body.get("speed"), 0.6), seed, n)
            return self._send(200, {"ok": True, "seed": seed})
        if GAME is None:
            return self._send(400, {"error": "no game"})
        if self.path == "/api/bid":
            err = GAME.answer("bid", body.get("bid"))
        elif self.path == "/api/play":
            err = GAME.answer("play", body.get("card"))
        elif self.path == "/api/next":
            err = GAME.answer("next", 0)
        else:
            return self._send(404, {"error": "not found"})
        self._send(400 if err else 200, {"error": err} if err else {"ok": True})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    srv = None
    for port in range(a.port, a.port + 20):                 # use the next free port if 8000 is taken
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    if srv is None:
        print("Could not open a local port."); input("Press Enter to close"); return
    url = f"http://localhost:{port}"
    print(f"Boerenbridge is running at {url}")
    print("Your browser opens automatically. Close this window (or press Ctrl+C) to quit the game.")
    if not a.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
