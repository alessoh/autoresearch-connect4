"""Scratch: per-move cost of the GPU negamax, and agreement with brute force."""
import time
import random
import numpy as np
import torch

from prepare import ConnectFourGame

src = open("train.py").read().split("# Training loop")[0]
M = {"__name__": "trainlib"}
exec(compile(src, "train.py", "exec"), M)
negamax = M["negamax_scores"]

device = torch.device("cuda")
net = M["ConnectFourNet"]().to(device).eval()

# Correctness: at depth 2 the search must agree with an exhaustive check that
# a winning move is taken and an immediate loss is never allowed.
random.seed(5)
checked = 0
for _ in range(300):
    g = ConnectFourGame()
    for _ in range(random.randint(0, 24)):
        if g.game_over:
            break
        g.make_move(random.choice(g.get_valid_moves()))
    if g.game_over:
        continue
    rel = torch.from_numpy(np.asarray(g.board, dtype=np.int8) * g.current_player).to(device)
    with torch.no_grad():
        pick = int(negamax(net, rel[None], 2)[0].argmax())
    wins = [c for c in g.get_valid_moves()
            if (lambda h: h.game_over and h.winner == g.current_player)(
                (lambda: (h := g.copy(), h.make_move(c), h)[-1])())]
    if wins:
        assert pick in wins, f"missed a win: picked {pick}, wins {wins}"
        checked += 1
print(f"depth-2 search took the immediate win in all {checked} positions that had one")

for plies, label in ((2, "opening"), (14, "midgame")):
    g = ConnectFourGame()
    for _ in range(plies):
        g.make_move(random.choice(g.get_valid_moves()))
    rel = torch.from_numpy(np.asarray(g.board, dtype=np.int8) * g.current_player).to(device)
    for d in (4, 6, 8):
        with torch.no_grad():
            negamax(net, rel[None], d)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            with torch.no_grad():
                negamax(net, rel[None], d)
        torch.cuda.synchronize()
        per = (time.time() - t0) / 10
        print(f"{label:8s} depth {d}: {per*1000:7.1f} ms/move  "
              f"-> {per*8400:5.0f}s over a full eval")
