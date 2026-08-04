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

# ---------------------------------------------------------------------------
# Neural Network
# ---------------------------------------------------------------------------

def four_in_a_row(p):
    """p: (n, 6, 7) bool mask of one player's pieces -> (n,) bool, has a four."""
    h = p[:, :, 0:4] & p[:, :, 1:5] & p[:, :, 2:6] & p[:, :, 3:7]
    v = p[:, 0:3] & p[:, 1:4] & p[:, 2:5] & p[:, 3:6]
    d = p[:, 0:3, 0:4] & p[:, 1:4, 1:5] & p[:, 2:5, 2:6] & p[:, 3:6, 3:7]
    u = p[:, 3:6, 0:4] & p[:, 2:5, 1:5] & p[:, 1:4, 2:6] & p[:, 0:3, 3:7]
    return (h.any(axis=(1, 2)) | v.any(axis=(1, 2))
            | d.any(axis=(1, 2)) | u.any(axis=(1, 2)))


def tactical_bias(boards):
    """
    Exact two-ply tactics, vectorized over a batch of positions.

    boards: (n, 6, 7) int8, mover-relative (+1 = player to move, -1 opponent).
    Returns (n, 7) float32 to add to the policy logits:
      +1e6  a move that wins immediately (play it)
      +1e5  the only reply to an opponent's immediate win (block it)
      -1e5  a move that lets the opponent win right on top of it
      -1e9  illegal column
    The policy still picks between equally-tactical moves, so this only
    removes the blunders a bare policy net keeps making.
    """
    n = boards.shape[0]
    heights = (boards != 0).sum(axis=1)                  # (n, 7)
    legal = heights < BOARD_ROWS
    landing = BOARD_ROWS - 1 - heights                   # (n, 7), <0 if full

    rep = np.repeat(boards, BOARD_COLS, axis=0)          # (n*7, 6, 7)
    flat = np.arange(n * BOARD_COLS)
    cols = np.tile(np.arange(BOARD_COLS), n)
    rows = np.clip(landing.reshape(-1), 0, BOARD_ROWS - 1)

    mine = rep.copy()
    mine[flat, rows, cols] = 1
    win_now = four_in_a_row(mine == 1).reshape(n, BOARD_COLS) & legal

    theirs = rep.copy()
    theirs[flat, rows, cols] = -1
    they_win = four_in_a_row(theirs == -1).reshape(n, BOARD_COLS) & legal

    # After my move the square above it opens up for them (win_now is already
    # computed, so `mine` can be reused in place)
    above = mine
    above[flat, np.clip(rows - 1, 0, BOARD_ROWS - 1), cols] = -1
    gives_win = (four_in_a_row(above == -1).reshape(n, BOARD_COLS)
                 & legal & (landing > 0))

    bias = np.where(legal, 0.0, -1e9).astype(np.float32)
    threatened = they_win.any(axis=1, keepdims=True)
    bias = np.where(threatened & they_win, 1e5, bias)
    bias = np.where(~threatened & gives_win, -1e5, bias)
    bias = np.where(win_now, 1e6, bias)
    return bias


class ConnectFourNet(nn.Module):
    """
    Policy-value network for Connect Four.
    Input: (batch, 2, 6, 7) — two channels (own pieces, opponent pieces).
    Output: (batch, 7) — logits over columns (policy head), with an exact
    two-ply tactical bias added so the greedy player never misses a win or
    an immediate block.

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

    def forward(self, x):
        """
        x: (batch, 2, 6, 7) float tensor.
        Returns: (batch, 7) tactically-corrected logits over columns.
        """
        logits = self.policy_logits(x)
        boards = (x[:, 0] - x[:, 1]).to(torch.int8).cpu().numpy()
        bias = torch.from_numpy(tactical_bias(boards)).to(logits.device)
        return logits + bias

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
            x = torch.from_numpy(encode_boards(boards, persps)).to(device)
            logits = model.policy_logits(x)

            # Play the same tactically-corrected policy that evaluation uses:
            # self-play games stay sharp instead of both sides overlooking
            # wins, which also makes the value targets meaningful. The bias
            # masks full columns too.
            rel = np.asarray(boards, dtype=np.int8) * np.asarray(persps, dtype=np.int8)[:, None, None]
            bias = torch.from_numpy(tactical_bias(rel)).to(device)
            cols = torch.multinomial(F.softmax(logits + bias, dim=1), 1).squeeze(1).tolist()

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
