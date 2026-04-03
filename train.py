"""
Connect Four Autoresearch training script. Single-GPU, single-file.
Modeled after Karpathy's Autoresearch train.py.
Usage: uv run train.py
"""

import os
import time
import random
from dataclasses import dataclass

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
LEARNING_RATE = 0.001       # optimizer learning rate
WEIGHT_DECAY = 1e-4         # L2 regularization
BATCH_SIZE = 64             # batch size for training updates
GAMMA = 0.99                # discount factor for returns
EXPLORATION_RATE = 0.15     # fraction of random moves during self-play
SELF_PLAY_RATIO = 1.0       # fraction of games that are self-play (rest vs opponents)
GAMES_PER_BATCH = 64        # games to play before each training update
DEVICE_BATCH_SIZE = 128     # max positions per forward pass during training

# ---------------------------------------------------------------------------
# Neural Network
# ---------------------------------------------------------------------------

class ConnectFourNet(nn.Module):
    """
    Policy-value network for Connect Four.
    Input: (batch, 2, 6, 7) — two channels (own pieces, opponent pieces).
    Output: (batch, 7) — logits over columns (policy head).
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

    def forward(self, x):
        """
        x: (batch, 2, 6, 7) float tensor.
        Returns: (batch, 7) logits over columns.
        """
        features = self.backbone(x)
        logits = self.policy_head(features)
        return logits


# ---------------------------------------------------------------------------
# Self-play data collection
# ---------------------------------------------------------------------------

def collect_game_data(model, device, exploration_rate=EXPLORATION_RATE):
    """
    Play one game of self-play and collect (state, action, reward) tuples
    from the perspective of each player.
    Returns list of (board_tensor, action, reward) for the winning side,
    and (board_tensor, action, -reward) for the losing side.
    """
    game = ConnectFourGame()
    history = []  # (board_tensor, action, player)

    model.eval()
    with torch.no_grad():
        while not game.game_over:
            player = game.current_player
            board_t = game.get_board_tensor(perspective=player)
            valid_moves = game.get_valid_moves()

            if random.random() < exploration_rate:
                # Random exploration
                col = random.choice(valid_moves)
            else:
                # Model choice
                x = board_t.unsqueeze(0).to(device)
                logits = model(x).squeeze(0)
                # Mask invalid moves
                mask = torch.full((BOARD_COLS,), -1e9, device=device)
                for v in valid_moves:
                    mask[v] = 0.0
                logits = logits + mask
                probs = F.softmax(logits, dim=0)
                col = torch.multinomial(probs, 1).item()

            history.append((board_t, col, player))
            game.make_move(col)

    # Assign rewards based on outcome
    data = []
    for board_t, action, player in history:
        if game.winner == player:
            reward = 1.0
        elif game.winner == -player:
            reward = -1.0
        else:
            reward = 0.0  # draw
        data.append((board_t, action, reward))

    return data


def collect_opponent_game_data(model, opponent, device, exploration_rate=EXPLORATION_RATE):
    """
    Play one game between the model and a fixed opponent.
    Collect training data only from the model's perspective.
    """
    game = ConnectFourGame()
    model_player = 1 if random.random() < 0.5 else -1
    history = []

    model.eval()
    with torch.no_grad():
        while not game.game_over:
            if game.current_player == model_player:
                board_t = game.get_board_tensor(perspective=model_player)
                valid_moves = game.get_valid_moves()

                if random.random() < exploration_rate:
                    col = random.choice(valid_moves)
                else:
                    x = board_t.unsqueeze(0).to(device)
                    logits = model(x).squeeze(0)
                    mask = torch.full((BOARD_COLS,), -1e9, device=device)
                    for v in valid_moves:
                        mask[v] = 0.0
                    logits = logits + mask
                    probs = F.softmax(logits, dim=0)
                    col = torch.multinomial(probs, 1).item()

                history.append((board_t, col))
                game.make_move(col)
            else:
                col = opponent.choose_move(game)
                game.make_move(col)

    # Assign rewards
    data = []
    for board_t, action in history:
        if game.winner == model_player:
            reward = 1.0
        elif game.winner == -model_player:
            reward = -1.0
        else:
            reward = 0.0
        data.append((board_t, action, reward))

    return data


def collect_batch(model, device, num_games=GAMES_PER_BATCH):
    """Collect a batch of training data from self-play and opponent play."""
    all_data = []
    num_self_play = int(num_games * SELF_PLAY_RATIO)
    num_opponent = num_games - num_self_play

    for _ in range(num_self_play):
        all_data.extend(collect_game_data(model, device))

    for _ in range(num_opponent):
        opp = random.choice(OPPONENTS[:2])  # play against easier opponents during training
        all_data.extend(collect_opponent_game_data(model, opp, device))

    return all_data


# ---------------------------------------------------------------------------
# Training step (REINFORCE with baseline)
# ---------------------------------------------------------------------------

def train_step(model, optimizer, batch_data, device):
    """
    One training step using REINFORCE policy gradient.
    batch_data: list of (board_tensor, action, reward)
    """
    if len(batch_data) == 0:
        return 0.0

    model.train()

    # Compute baseline (mean reward)
    rewards = [d[2] for d in batch_data]
    baseline = sum(rewards) / len(rewards)

    # Process in mini-batches to manage memory
    total_loss = 0.0
    num_samples = 0

    for i in range(0, len(batch_data), DEVICE_BATCH_SIZE):
        chunk = batch_data[i:i + DEVICE_BATCH_SIZE]
        boards = torch.stack([d[0] for d in chunk]).to(device)
        actions = torch.tensor([d[1] for d in chunk], dtype=torch.long, device=device)
        rews = torch.tensor([d[2] for d in chunk], dtype=torch.float32, device=device)

        logits = model(boards)
        log_probs = F.log_softmax(logits, dim=1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        advantage = rews - baseline
        loss = -(action_log_probs * advantage).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(chunk)
        num_samples += len(chunk)

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
