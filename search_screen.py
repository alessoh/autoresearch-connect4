"""
Scratch: score a saved checkpoint on the official metric at several search
depths, so evaluation-side ideas need no retraining.

Usage: CKPT=model.pt uv run search_screen.py [depth ...]
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import evaluate_winrate

src = open("train.py").read().split("# Training loop")[0]
M = {"__name__": "trainlib"}
exec(compile(src, "train.py", "exec"), M)
negamax_scores = M["negamax_scores"]


class SearchPlayer(nn.Module):
    """The evaluation player, with the search depth overridden."""

    def __init__(self, net, depth):
        super().__init__()
        self.net, self.depth = net, depth

    @torch.no_grad()
    def forward(self, x):
        boards = (x[:, 0] - x[:, 1]).to(torch.int8)
        scores = negamax_scores(self.net, boards, self.depth)
        prior = F.softmax(self.net.policy_logits(x), dim=1)
        return scores + M["SEARCH_PRIOR"] * prior


device = torch.device("cuda")
net = M["ConnectFourNet"]().to(device)
net.load_state_dict(torch.load(os.environ.get("CKPT", "model.pt"), map_location=device))
net.eval()

for depth in ([int(a) for a in sys.argv[1:]] or [M["SEARCH_DEPTH"]]):
    t0 = time.time()
    r = evaluate_winrate(SearchPlayer(net, depth).to(device).eval(), device=str(device))
    per = "  ".join(f"{k}={v['win_rate']:.2f}" for k, v in r["per_opponent"].items())
    print(f"depth {depth}: win_rate={r['win_rate']:.6f}  [{per}]  ({time.time()-t0:.0f}s)",
          flush=True)
