"""
Connect Four Autoresearch training script. Single-GPU, single-file.
Modeled after Karpathy's Autoresearch train.py.
Usage: uv run train.py
"""

import os
import time
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from prepare import (
    BOARD_ROWS, BOARD_COLS, TIME_BUDGET,
    ConnectFourGame, OPPONENTS, evaluate_winrate,
)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
NUM_CONV_LAYERS = 3         # number of convolutional layers
CONV_CHANNELS = 64          # channels per conv layer
FC_HIDDEN = 128             # hidden units in fully connected layer

# Training
LEARNING_RATE = 0.0005      # optimizer learning rate
WEIGHT_DECAY = 1e-4         # L2 regularization
BATCH_SIZE = 64             # batch size for training updates
GAMMA = 0.99                # discount factor for returns
EXPLORATION_RATE = 0.15     # fraction of random moves during self-play
SELF_PLAY_RATIO = 0.7       # fraction of games that are self-play (rest vs opponents)
ENTROPY_COEF = 0.03         # entropy bonus to prevent policy collapse
VALUE_COEF = 0.5            # weight of the value-head regression loss
GAMES_PER_BATCH = 64        # games to play before each training update
DEVICE_BATCH_SIZE = 128     # max positions per forward pass during training

# Evaluation-time search (must be even: an odd depth leaves the leaves scored
# right after our own move, which makes the search optimistic and much weaker)
SEARCH_DEPTH = 6            # plies of full-width negamax when playing a move
SEARCH_DISCOUNT = 0.99      # per-ply discount, so faster wins score higher
SEARCH_PRIOR = 1e-3         # weight of the policy as a tiebreak between equals
SEARCH_CHUNK = 65536        # leaves scored per value-head call (caps VRAM)
SELF_PLAY_DEPTH = 2         # search depth used to play the training games
SELF_PLAY_TEMP = 0.2        # temperature for sampling from the search scores

# ---------------------------------------------------------------------------
# Neural Network
# ---------------------------------------------------------------------------

def four_in_a_row(p):
    """
    p: (n, 6, 7) bool mask of one player's pieces -> (n,) bool, has a four.
    Written so it runs unchanged on numpy arrays (self-play, on CPU) and on
    torch tensors (evaluation search, on GPU).
    """
    h = p[:, :, 0:4] & p[:, :, 1:5] & p[:, :, 2:6] & p[:, :, 3:7]
    v = p[:, 0:3] & p[:, 1:4] & p[:, 2:5] & p[:, 3:6]
    d = p[:, 0:3, 0:4] & p[:, 1:4, 1:5] & p[:, 2:5, 2:6] & p[:, 3:6, 3:7]
    u = p[:, 3:6, 0:4] & p[:, 2:5, 1:5] & p[:, 1:4, 2:6] & p[:, 0:3, 3:7]
    return (h.any(-1).any(-1) | v.any(-1).any(-1)
            | d.any(-1).any(-1) | u.any(-1).any(-1))


LIVE, LOST, DRAW = 0, 1, 2   # node status after the move that created it


def expand(boards):
    """
    One ply of full-width expansion, vectorized on whatever device `boards`
    lives on.
    boards: (n, 6, 7) int8 mover-relative, all non-terminal.
    Returns (children, parent_index, column, status); children are again
    mover-relative, and LOST means the child's mover has already lost.
    """
    heights = (boards != 0).sum(1)
    node_idx, col_idx = torch.nonzero(heights < BOARD_ROWS, as_tuple=True)
    child = boards[node_idx].clone()
    rows = BOARD_ROWS - 1 - heights[node_idx, col_idx]
    child[torch.arange(child.shape[0], device=boards.device), rows, col_idx] = 1
    win = four_in_a_row(child == 1)
    full = (child != 0).all(-1).all(-1)
    status = torch.where(win, LOST, torch.where(full, DRAW, LIVE))
    return -child, node_idx, col_idx, status


@torch.no_grad()
def negamax_scores(model, roots, depth):
    """
    Full-width negamax from many root positions at once, kept entirely on the
    GPU: one ply is one batch of tensor ops, and every leaf at the final ply is
    scored in one value-head call. Terminal nodes are scored exactly, so the
    search never misses a win or a forced loss inside the horizon.

    roots: (n, 6, 7) int8 tensor, mover-relative (+1 = player to move), all
    non-terminal.
    Returns (n, 7) move scores from each root mover's view, -inf for illegal.
    """
    device = roots.device
    levels = [(roots, None, None, torch.zeros(len(roots), dtype=torch.long,
                                              device=device))]
    for _ in range(depth):
        pb, _, _, pstat = levels[-1]
        live = torch.nonzero(pstat == LIVE, as_tuple=True)[0]
        child, par, col, cstat = expand(pb[live])
        levels.append((child, live[par], col, cstat))

    # Deepest ply: exact for finished games, value head for the rest
    boards, _, _, status = levels[depth]
    values = torch.where(status == LOST, -1.0, 0.0)
    live = torch.nonzero(status == LIVE, as_tuple=True)[0]
    for i in range(0, live.numel(), SEARCH_CHUNK):    # chunked to bound VRAM
        idx = live[i:i + SEARCH_CHUNK]
        b = boards[idx]
        values[idx] = model.value(torch.stack((b == 1, b == -1), dim=1).float()).float()

    # Negamax back-up, stopping at ply 1 to keep one score per root move
    for d in range(depth - 1, 0, -1):
        _, _, _, pstat = levels[d]
        _, par, _, _ = levels[d + 1]
        best = torch.full((pstat.shape[0],), -float("inf"), device=device)
        best = best.scatter_reduce(0, par, -SEARCH_DISCOUNT * values, "amax")
        values = torch.where(pstat == LOST, -1.0,
                             torch.where(pstat == DRAW, 0.0, best))

    _, root_idx, col, _ = levels[1]
    scores = torch.full((len(roots), BOARD_COLS), -float("inf"), device=device)
    scores[root_idx, col] = -SEARCH_DISCOUNT * values
    return scores


class ConnectFourNet(nn.Module):
    """
    Policy-value network for Connect Four.
    Input: (batch, 2, 6, 7) — two channels (own pieces, opponent pieces).
    Output: (batch, 7) — negamax scores over columns, so the greedy player of
    prepare.play_game plays the search rather than the bare policy.

    The value head predicts the game outcome in [-1, 1] from the perspective
    of the player to move. forward() deliberately returns only the policy
    logits, because prepare.play_game expects a plain (batch, 7) tensor.
    """

    def __init__(self):
        super().__init__()

        # Convolutional backbone
        layers = []
        in_channels = 2
        for i in range(NUM_CONV_LAYERS):
            layers.append(nn.Conv2d(in_channels, CONV_CHANNELS, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(CONV_CHANNELS))
            layers.append(nn.ReLU())
            in_channels = CONV_CHANNELS
        self.backbone = nn.Sequential(*layers)

        # Policy head
        flat_size = CONV_CHANNELS * BOARD_ROWS * BOARD_COLS
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, FC_HIDDEN),
            nn.ReLU(),
            nn.Linear(FC_HIDDEN, BOARD_COLS),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, FC_HIDDEN),
            nn.ReLU(),
            nn.Linear(FC_HIDDEN, 1),
            nn.Tanh(),
        )

    @torch.no_grad()
    def forward(self, x):
        """
        x: (batch, 2, 6, 7) float tensor. Returns (batch, 7) negamax scores,
        with the policy mixed in at low weight to break ties between equals.
        """
        boards = (x[:, 0] - x[:, 1]).to(torch.int8)
        scores = negamax_scores(self, boards, SEARCH_DEPTH)
        return scores + SEARCH_PRIOR * F.softmax(self.policy_logits(x), dim=1)

    def policy_logits(self, x):
        """Raw policy logits, no tactical correction (used during self-play)."""
        return self.policy_head(self.backbone(x))

    def policy_value(self, x):
        """Returns ((batch, 7) logits, (batch,) values in [-1, 1])."""
        features = self.backbone(x)
        return self.policy_head(features), self.value_head(features).squeeze(1)

    def value(self, x):
        """Returns (batch,) values in [-1, 1] for the player to move."""
        return self.value_head(self.backbone(x)).squeeze(1)


# ---------------------------------------------------------------------------
# Self-play data collection
# ---------------------------------------------------------------------------

def encode_boards(boards, perspectives):
    """
    Vectorized board encoder — the batched equivalent of get_board_tensor.
    boards: sequence of BOARD_ROWS x BOARD_COLS nested lists (or arrays).
    perspectives: sequence of +1/-1, one per board.
    Returns (n, 2, BOARD_ROWS, BOARD_COLS) float32 array: own pieces, then
    opponent pieces.
    """
    b = np.asarray(boards, dtype=np.int8).reshape(-1, BOARD_ROWS, BOARD_COLS)
    p = np.asarray(perspectives, dtype=np.int8).reshape(-1, 1, 1)
    x = np.empty((b.shape[0], 2, BOARD_ROWS, BOARD_COLS), dtype=np.float32)
    x[:, 0] = b == p
    x[:, 1] = b == -p
    return x


def collect_batch(model, device, num_games=GAMES_PER_BATCH):
    """
    Play num_games concurrently, batching every network forward into one call
    per ply instead of one call per move. A fraction SELF_PLAY_RATIO of the
    games are self-play; the rest are against the easy fixed opponents.
    Returns (states, actions, rewards) as numpy arrays.
    """
    games = [ConnectFourGame() for _ in range(num_games)]
    num_self_play = int(num_games * SELF_PLAY_RATIO)

    # model_side[i] is None for self-play (the model moves for both sides).
    model_side = [None] * num_self_play
    opponents = [None] * num_self_play
    for _ in range(num_games - num_self_play):
        model_side.append(1 if random.random() < 0.5 else -1)
        opponents.append(random.choice(OPPONENTS[:2]))  # easier opponents while training

    hist_board, hist_persp, hist_action, hist_game = [], [], [], []

    model.eval()
    with torch.no_grad():
        while True:
            # Fixed opponents move first: they are pure Python and need no batching.
            for i, g in enumerate(games):
                if model_side[i] is not None and not g.game_over \
                        and g.current_player != model_side[i]:
                    g.make_move(opponents[i].choose_move(g))

            active = [i for i, g in enumerate(games) if not g.game_over]
            if not active:
                break

            boards = [games[i].board for i in active]
            persps = [games[i].current_player for i in active]

            # Self-play moves come from the same search that plays evaluation,
            # just shallower: one batched negamax covers every live game. The
            # games stay sharp instead of both sides overlooking wins, so the
            # outcomes the value head regresses are meaningful. Sampling at a
            # low temperature keeps the games varied.
            rel = np.asarray(boards, dtype=np.int8) * np.asarray(persps, dtype=np.int8)[:, None, None]
            scores = negamax_scores(model, torch.from_numpy(rel).to(device),
                                    SELF_PLAY_DEPTH)
            cols = torch.multinomial(F.softmax(scores / SELF_PLAY_TEMP, dim=1),
                                     1).squeeze(1).tolist()

            for j, i in enumerate(active):
                g = games[i]
                col = cols[j]
                if random.random() < EXPLORATION_RATE:
                    col = random.choice(g.get_valid_moves())
                hist_board.append(np.asarray(g.board, dtype=np.int8))
                hist_persp.append(g.current_player)
                hist_action.append(col)
                hist_game.append(i)
                g.make_move(col)

    # Terminal reward, from the perspective of whoever was to move
    winners = np.array([g.winner for g in games], dtype=np.int8)[hist_game]
    persps = np.array(hist_persp, dtype=np.int8)
    rewards = np.where(winners == persps, 1.0,
                       np.where(winners == -persps, -1.0, 0.0)).astype(np.float32)

    states = encode_boards(hist_board, persps)
    return states, np.array(hist_action, dtype=np.int64), rewards


# ---------------------------------------------------------------------------
# Training step (REINFORCE with baseline)
# ---------------------------------------------------------------------------

def train_step(model, optimizer, batch_data, device):
    """
    One training step using REINFORCE policy gradient.
    batch_data: (states, actions, rewards) numpy arrays.
    """
    states, all_actions, all_rewards = batch_data
    num_positions = len(all_actions)
    if num_positions == 0:
        return 0.0

    model.train()

    # Scale for the advantage, computed over the whole collection
    adv_std = float(all_rewards.std()) + 1e-8  # prevent division by zero

    # Process in mini-batches to manage memory
    total_loss = 0.0
    num_samples = 0

    for i in range(0, num_positions, DEVICE_BATCH_SIZE):
        sl = slice(i, i + DEVICE_BATCH_SIZE)
        boards = torch.from_numpy(states[sl]).to(device)
        actions = torch.from_numpy(all_actions[sl]).to(device)
        rews = torch.from_numpy(all_rewards[sl]).to(device)
        chunk_size = len(actions)

        logits, values = model.policy_value(boards)
        log_probs = F.log_softmax(logits, dim=1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Clip log_probs: exploration forces moves the model hates, creating
        # extreme log_probs (e.g. -15). Clamping prevents any single sample
        # from dominating the gradient.
        action_log_probs = action_log_probs.clamp(min=-5.0)

        # Advantage against the learned value baseline, normalized and clipped
        advantage = (rews - values.detach()) / adv_std
        advantage = advantage.clamp(-3.0, 3.0)

        # Policy gradient loss
        pg_loss = -(action_log_probs * advantage).mean()

        # Entropy bonus: prevents policy from collapsing to a single action
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()

        # Value head regresses the game outcome from the mover's perspective
        value_loss = F.mse_loss(values, rews)

        loss = pg_loss - ENTROPY_COEF * entropy + VALUE_COEF * value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * chunk_size
        num_samples += chunk_size

    return total_loss / max(num_samples, 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Build model
model = ConnectFourNet().to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

# Optimizer
optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

# Training
print(f"Time budget: {TIME_BUDGET}s")
print(f"Self-play ratio: {SELF_PLAY_RATIO}")
print(f"Exploration rate: {EXPLORATION_RATE}")
print(f"Entropy coef: {ENTROPY_COEF}")
print(f"Games per batch: {GAMES_PER_BATCH}")
print()

t_start_training = time.time()
total_training_time = 0.0
step = 0
total_games = 0
smooth_loss = 0.0

while True:
    t0 = time.time()

    # Collect self-play data
    batch_data = collect_batch(model, device)
    total_games += GAMES_PER_BATCH

    # Train on collected data
    loss = train_step(model, optimizer, batch_data, device)

    # Smooth loss for logging
    if step == 0:
        smooth_loss = loss
    else:
        smooth_loss = 0.9 * smooth_loss + 0.1 * loss

    step += 1
    elapsed = time.time() - t0
    total_training_time = time.time() - t_start_training

    # Log every 10 steps
    if step % 10 == 0:
        print(f"step {step:4d} | loss {smooth_loss:.4f} | games {total_games:5d} | "
              f"time {total_training_time:.0f}s/{TIME_BUDGET}s")

    # Check time budget
    if total_training_time >= TIME_BUDGET:
        break

print(f"\nTraining complete: {step} steps, {total_games} games, "
      f"{total_training_time:.1f}s")

# Checkpoint, so evaluation-time ideas can be tried without retraining
torch.save(model.state_dict(), "model.pt")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

print("\nEvaluating against opponent suite...")
t_eval_start = time.time()
results = evaluate_winrate(model, device=str(device))
t_eval = time.time() - t_eval_start

print(f"\nPer-opponent results:")
for name, stats in results["per_opponent"].items():
    wr = stats["win_rate"]
    w, l, d = stats["wins"], stats["losses"], stats["draws"]
    print(f"  {name:15s}: {wr:.3f} win_rate ({w}W {l}L {d}D / {stats['games']})")

t_total = time.time() - t_start
peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

print(f"\n---")
print(f"win_rate:          {results['win_rate']:.6f}")
print(f"training_seconds:  {total_training_time:.1f}")
print(f"total_seconds:     {t_total:.1f}")
print(f"eval_seconds:      {t_eval:.1f}")
print(f"peak_vram_mb:      {peak_vram_mb:.1f}")
print(f"total_games:       {total_games}")
print(f"num_steps:         {step}")
print(f"num_params:        {num_params}")
print(f"conv_layers:       {NUM_CONV_LAYERS}")
print(f"conv_channels:     {CONV_CHANNELS}")
print(f"fc_hidden:         {FC_HIDDEN}")
print(f"entropy_coef:      {ENTROPY_COEF}")
