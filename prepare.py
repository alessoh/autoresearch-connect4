"""
Immutable harness for Connect Four Autoresearch.
Contains the game engine, fixed opponents, and the sacred evaluation metric.

Usage:
    python prepare.py              # verify setup, run self-test
    python prepare.py --benchmark  # benchmark opponent speeds

DO NOT MODIFY THIS FILE. It is the equivalent of Karpathy's prepare.py.
The evaluate_winrate function is the ground truth metric.
"""

import os
import time
import math
import random
import argparse
from typing import Optional

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

BOARD_ROWS = 6
BOARD_COLS = 7
WIN_LENGTH = 4
TIME_BUDGET = 300          # training time budget in seconds (5 minutes)
EVAL_GAMES = 100           # games per opponent during evaluation
CONNECT = 4                # four in a row to win

# ---------------------------------------------------------------------------
# Game Engine
# ---------------------------------------------------------------------------

class ConnectFourGame:
    """
    Connect Four game engine. Board is stored as a 2D list (row 0 = top).
    Players are 1 and -1. 0 means empty.
    """

    def __init__(self):
        self.board = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]
        self.current_player = 1
        self.move_count = 0
        self.last_move = None
        self.game_over = False
        self.winner = 0  # 0 = no winner / draw, 1 or -1 = that player won

    def copy(self):
        """Return a deep copy of the game state."""
        g = ConnectFourGame()
        g.board = [row[:] for row in self.board]
        g.current_player = self.current_player
        g.move_count = self.move_count
        g.last_move = self.last_move
        g.game_over = self.game_over
        g.winner = self.winner
        return g

    def get_valid_moves(self):
        """Return list of columns that are not full."""
        return [c for c in range(BOARD_COLS) if self.board[0][c] == 0]

    def make_move(self, col):
        """
        Drop a disc in the given column. Returns the row where it landed.
        Raises ValueError if column is full or game is over.
        """
        if self.game_over:
            raise ValueError("Game is already over")
        if col < 0 or col >= BOARD_COLS or self.board[0][col] != 0:
            raise ValueError(f"Invalid move: column {col}")

        # Find the lowest empty row in this column
        row = BOARD_ROWS - 1
        while row >= 0 and self.board[row][col] != 0:
            row -= 1

        self.board[row][col] = self.current_player
        self.last_move = (row, col)
        self.move_count += 1

        # Check for win
        if self._check_win(row, col):
            self.game_over = True
            self.winner = self.current_player
        elif self.move_count >= BOARD_ROWS * BOARD_COLS:
            self.game_over = True
            self.winner = 0  # draw

        self.current_player *= -1
        return row

    def _check_win(self, row, col):
        """Check if the last move at (row, col) creates four in a row."""
        player = self.board[row][col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1
            # Count in positive direction
            r, c = row + dr, col + dc
            while 0 <= r < BOARD_ROWS and 0 <= c < BOARD_COLS and self.board[r][c] == player:
                count += 1
                r += dr
                c += dc
            # Count in negative direction
            r, c = row - dr, col - dc
            while 0 <= r < BOARD_ROWS and 0 <= c < BOARD_COLS and self.board[r][c] == player:
                count += 1
                r -= dr
                c -= dc
            if count >= WIN_LENGTH:
                return True
        return False

    def get_board_tensor(self, perspective=None):
        """
        Return board as a (2, BOARD_ROWS, BOARD_COLS) float tensor.
        Channel 0 = current player's pieces, Channel 1 = opponent's pieces.
        If perspective is given, encode from that player's perspective.
        """
        if perspective is None:
            perspective = self.current_player
        t = torch.zeros(2, BOARD_ROWS, BOARD_COLS)
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                if self.board[r][c] == perspective:
                    t[0, r, c] = 1.0
                elif self.board[r][c] == -perspective:
                    t[1, r, c] = 1.0
        return t

    def get_board_flat(self):
        """Return board as a flat list of BOARD_ROWS * BOARD_COLS values."""
        flat = []
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                flat.append(self.board[r][c])
        return flat

    def render(self):
        """Return a string representation of the board."""
        symbols = {0: '.', 1: 'X', -1: 'O'}
        lines = []
        for row in self.board:
            lines.append(' '.join(symbols[cell] for cell in row))
        lines.append(' '.join(str(c) for c in range(BOARD_COLS)))
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Fixed Opponents (DO NOT MODIFY)
# ---------------------------------------------------------------------------

class RandomOpponent:
    """Plays a uniformly random legal move."""
    name = "random"

    def choose_move(self, game):
        moves = game.get_valid_moves()
        return random.choice(moves)


class OneStepOpponent:
    """Wins if it can, blocks opponent wins, otherwise plays center-biased random."""
    name = "one_step"

    def choose_move(self, game):
        moves = game.get_valid_moves()

        # Check if we can win immediately
        for col in moves:
            g = game.copy()
            g.make_move(col)
            if g.game_over and g.winner == game.current_player:
                return col

        # Check if opponent would win next turn, and block
        for col in moves:
            g = game.copy()
            g.current_player *= -1  # pretend opponent plays
            g.make_move(col)
            if g.game_over and g.winner != 0:
                return col

        # Prefer center columns
        center = BOARD_COLS // 2
        preference = sorted(moves, key=lambda c: abs(c - center))
        return preference[0]


class MinimaxOpponent:
    """Minimax with alpha-beta pruning and configurable depth."""

    def __init__(self, depth=3):
        self.depth = depth
        self.name = f"minimax_d{depth}"

    def choose_move(self, game):
        moves = game.get_valid_moves()
        if not moves:
            return None

        best_score = -math.inf
        best_move = moves[0]

        for col in moves:
            g = game.copy()
            g.make_move(col)
            score = self._minimax(g, self.depth - 1, -math.inf, math.inf, False,
                                  game.current_player)
            if score > best_score:
                best_score = score
                best_move = col

        return best_move

    def _minimax(self, game, depth, alpha, beta, maximizing, original_player):
        if game.game_over:
            if game.winner == original_player:
                return 1000 + depth  # prefer faster wins
            elif game.winner == -original_player:
                return -1000 - depth  # prefer slower losses
            else:
                return 0  # draw

        if depth == 0:
            return self._heuristic(game, original_player)

        moves = game.get_valid_moves()
        # Order moves: center columns first for better pruning
        center = BOARD_COLS // 2
        moves.sort(key=lambda c: abs(c - center))

        if maximizing:
            value = -math.inf
            for col in moves:
                g = game.copy()
                g.make_move(col)
                value = max(value, self._minimax(g, depth - 1, alpha, beta, False,
                                                 original_player))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = math.inf
            for col in moves:
                g = game.copy()
                g.make_move(col)
                value = min(value, self._minimax(g, depth - 1, alpha, beta, True,
                                                 original_player))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    def _heuristic(self, game, player):
        """Simple heuristic: count potential winning lines weighted by pieces."""
        score = 0
        board = game.board

        # Check all possible four-in-a-row windows
        # Horizontal
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS - 3):
                window = [board[r][c + i] for i in range(4)]
                score += self._score_window(window, player)
        # Vertical
        for r in range(BOARD_ROWS - 3):
            for c in range(BOARD_COLS):
                window = [board[r + i][c] for i in range(4)]
                score += self._score_window(window, player)
        # Diagonal down-right
        for r in range(BOARD_ROWS - 3):
            for c in range(BOARD_COLS - 3):
                window = [board[r + i][c + i] for i in range(4)]
                score += self._score_window(window, player)
        # Diagonal up-right
        for r in range(3, BOARD_ROWS):
            for c in range(BOARD_COLS - 3):
                window = [board[r - i][c + i] for i in range(4)]
                score += self._score_window(window, player)

        # Center column bonus
        center = BOARD_COLS // 2
        center_count = sum(1 for r in range(BOARD_ROWS) if board[r][center] == player)
        score += center_count * 3

        return score

    def _score_window(self, window, player):
        opp = -player
        p_count = window.count(player)
        o_count = window.count(opp)
        empty = window.count(0)

        if p_count == 4:
            return 100
        elif p_count == 3 and empty == 1:
            return 5
        elif p_count == 2 and empty == 2:
            return 2
        elif o_count == 3 and empty == 1:
            return -4
        return 0


# The fixed opponent suite — order matters for reporting
OPPONENTS = [
    RandomOpponent(),
    OneStepOpponent(),
    MinimaxOpponent(depth=3),
    MinimaxOpponent(depth=5),
]

# ---------------------------------------------------------------------------
# Play a single game: model vs opponent
# ---------------------------------------------------------------------------

def play_game(model, opponent, model_plays_first, device="cuda"):
    """
    Play one game between the model and an opponent.
    The model receives a board tensor and outputs logits over 7 columns.
    Returns: 1 if model wins, -1 if model loses, 0 if draw.
    """
    game = ConnectFourGame()
    model_player = 1 if model_plays_first else -1

    model.eval()
    with torch.no_grad():
        while not game.game_over:
            if game.current_player == model_player:
                # Model's turn
                board_t = game.get_board_tensor(perspective=model_player).unsqueeze(0).to(device)
                logits = model(board_t)  # shape (1, 7)
                # Mask invalid moves
                valid = game.get_valid_moves()
                mask = torch.full((BOARD_COLS,), -1e9, device=device)
                for v in valid:
                    mask[v] = 0.0
                logits = logits.squeeze(0) + mask
                col = logits.argmax().item()
            else:
                # Opponent's turn
                col = opponent.choose_move(game)

            game.make_move(col)

    if game.winner == model_player:
        return 1
    elif game.winner == -model_player:
        return -1
    else:
        return 0


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_winrate(model, device="cuda", num_games=EVAL_GAMES, seed=42):
    """
    Win rate evaluation: the sacred metric.

    Plays num_games against each opponent in OPPONENTS, alternating who goes
    first. Reports per-opponent win rate and overall weighted win rate.
    Uses a fixed seed for reproducibility.

    Returns a dict with:
        - per_opponent: dict mapping opponent name to win rate
        - win_rate: overall weighted win rate (0.0 to 1.0), higher is better
        - total_wins, total_losses, total_draws, total_games
    """
    rng_state = random.getstate()
    random.seed(seed)

    model.eval()
    total_wins = 0
    total_losses = 0
    total_draws = 0
    total_games = 0
    per_opponent = {}

    # Weight harder opponents more so the metric keeps discriminating
    # as the agent improves
    weights = {
        "random": 0.5,
        "one_step": 1.0,
        "minimax_d3": 2.0,
        "minimax_d5": 3.0,
    }

    weighted_sum = 0.0
    weight_total = 0.0

    for opp in OPPONENTS:
        wins = 0
        losses = 0
        draws = 0
        for i in range(num_games):
            model_first = (i % 2 == 0)
            result = play_game(model, opp, model_first, device=device)
            if result == 1:
                wins += 1
            elif result == -1:
                losses += 1
            else:
                draws += 1

        opp_winrate = wins / num_games
        per_opponent[opp.name] = {
            "wins": wins, "losses": losses, "draws": draws,
            "games": num_games, "win_rate": opp_winrate,
        }
        total_wins += wins
        total_losses += losses
        total_draws += draws
        total_games += num_games

        w = weights.get(opp.name, 1.0)
        weighted_sum += opp_winrate * w
        weight_total += w

    overall = weighted_sum / weight_total if weight_total > 0 else 0.0

    random.setstate(rng_state)

    return {
        "win_rate": overall,
        "per_opponent": per_opponent,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "total_draws": total_draws,
        "total_games": total_games,
    }


# ---------------------------------------------------------------------------
# Self-test and benchmark
# ---------------------------------------------------------------------------

def self_test():
    """Verify the game engine and opponents work correctly."""
    print("Running self-tests...")

    # Test basic game mechanics
    g = ConnectFourGame()
    assert len(g.get_valid_moves()) == 7
    g.make_move(3)
    assert g.board[5][3] == 1
    assert g.current_player == -1
    g.make_move(3)
    assert g.board[4][3] == -1
    print("  Game engine: OK")

    # Test win detection (vertical)
    g2 = ConnectFourGame()
    for _ in range(3):
        g2.make_move(0)  # player 1
        g2.make_move(1)  # player -1
    g2.make_move(0)  # player 1 wins with vertical four
    assert g2.game_over and g2.winner == 1
    print("  Win detection (vertical): OK")

    # Test win detection (horizontal)
    g3 = ConnectFourGame()
    for c in range(3):
        g3.make_move(c)      # player 1
        g3.make_move(c)      # player -1 (stacks on top)
    g3.make_move(3)           # player 1: four across bottom
    assert g3.game_over and g3.winner == 1
    print("  Win detection (horizontal): OK")

    # Test board tensor
    g4 = ConnectFourGame()
    g4.make_move(3)
    t = g4.get_board_tensor(perspective=1)
    assert t.shape == (2, BOARD_ROWS, BOARD_COLS)
    assert t[0, 5, 3] == 1.0  # player 1's piece
    print("  Board tensor: OK")

    # Test each opponent can play a full game without crashing
    for opp in OPPONENTS:
        g = ConnectFourGame()
        while not g.game_over:
            col = opp.choose_move(g)
            g.make_move(col)
        print(f"  Opponent '{opp.name}': OK (game ended in {g.move_count} moves)")

    print("\nAll self-tests passed.")


def benchmark_opponents():
    """Time how long each opponent takes to play a game."""
    print("Benchmarking opponents (10 games each)...\n")
    for opp in OPPONENTS:
        times = []
        for _ in range(10):
            g = ConnectFourGame()
            t0 = time.time()
            while not g.game_over:
                col = opp.choose_move(g)
                g.make_move(col)
            times.append(time.time() - t0)
        avg = sum(times) / len(times)
        print(f"  {opp.name:15s}: {avg*1000:.1f} ms/game")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Connect Four Autoresearch: setup and verification")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark opponent speeds")
    args = parser.parse_args()

    self_test()
    print()

    if args.benchmark:
        benchmark_opponents()

    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Eval games per opponent: {EVAL_GAMES}")
    print(f"Opponent suite: {[o.name for o in OPPONENTS]}")
    print(f"\nReady to train. Run: uv run train.py")
