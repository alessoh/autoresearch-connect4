"""
Scratch tool: low-variance strength meter for a saved checkpoint.

The official metric plays only 2 distinct games per (deterministic) minimax
opponent, so it is quantized to {0, .5, 1} and swings +-0.23 between identical
runs. This plays many games from randomized openings instead, giving a strength
signal fine enough to compare ideas.

Usage: CKPT=model.pt uv run arena.py [depth ...]
"""
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn.functional as F

from prepare import BOARD_COLS, ConnectFourGame, OPPONENTS

src = open("train.py").read().split("# Training loop")[0]
M = {"__name__": "trainlib"}
exec(compile(src, "train.py", "exec"), M)
encode_boards = M["encode_boards"]
negamax_scores = M["negamax_scores"]

OPENING_PLIES = 4                                   # random plies before the match
NUM_OPENINGS = int(os.environ.get("OPENINGS", 25))   # each played from both sides


def choose_move(net, game, depth, device):
    """Pick a move the way evaluation would: depth 0 = bare policy."""
    rel = np.asarray(game.board, dtype=np.int8) * game.current_player
    if depth == 0:
        x = torch.from_numpy(encode_boards([game.board], [game.current_player])).to(device)
        with torch.no_grad():
            logits = net.policy_logits(x).squeeze(0).cpu().numpy()
        mask = np.where(np.asarray(game.board[0]) == 0, 0.0, -np.inf)
        return int(np.argmax(logits + mask))
    scores = negamax_scores(net, torch.from_numpy(rel[None]).to(device), depth)
    return int(scores[0].argmax())


def openings(n, plies, seed=1234):
    """n random legal opening positions, reproducible across calls."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        g = ConnectFourGame()
        for _ in range(plies):
            g.make_move(rng.choice(g.get_valid_moves()))
        if not g.game_over:
            out.append([row[:] for row in g.board])
    return out


def match(net, opponent, depth, device, boards):
    """Play every opening from both sides. Returns (wins, losses, draws)."""
    w = l = d = 0
    for board in boards:
        for model_player in (1, -1):
            g = ConnectFourGame()
            g.board = [row[:] for row in board]
            g.move_count = sum(1 for r in g.board for c in r if c != 0)
            # openings are an even number of plies, so player 1 is to move
            g.current_player = 1
            while not g.game_over:
                if g.current_player == model_player:
                    g.make_move(choose_move(net, g, depth, device))
                else:
                    g.make_move(opponent.choose_move(g))
            if g.winner == model_player:
                w += 1
            elif g.winner == -model_player:
                l += 1
            else:
                d += 1
    return w, l, d


device = torch.device("cuda")
net = M["ConnectFourNet"]().to(device)
net.load_state_dict(torch.load(os.environ.get("CKPT", "model.pt"), map_location=device))
net.eval()

boards = openings(NUM_OPENINGS, OPENING_PLIES)
depths = [int(a) for a in sys.argv[1:]] or [0]
want = os.environ.get("OPPS", "one_step,minimax_d3,minimax_d5").split(",")
targets = [o for o in OPPONENTS if o.name in want]

print(f"{NUM_OPENINGS} openings x 2 sides = {2*NUM_OPENINGS} games per opponent")
for depth in depths:
    label = "raw policy" if depth == 0 else f"depth {depth}"
    parts, t0 = [], time.time()
    for opp in targets:
        w, l, d = match(net, opp, depth, device, boards)
        parts.append(f"{opp.name}={w/(2*NUM_OPENINGS):.2f}")
    print(f"  {label:11s} " + "  ".join(parts) + f"   ({time.time()-t0:.0f}s)", flush=True)
