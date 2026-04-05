# autoresearch-connect4

Autonomous game-playing agent architecture search for Connect Four, modeled directly on [Karpathy's Autoresearch](https://github.com/karpathy/autoresearch).

The idea: give an AI agent a Connect Four training setup and let it experiment autonomously overnight. It modifies the neural network code, trains for 5 minutes via self-play, checks if the agent plays better, keeps or discards the change, and repeats. You wake up in the morning to a log of experiments and a better game-playing agent.

This project adapts the Autoresearch pattern from LLM pretraining to reinforcement learning. The human writes `program.md` once, then sleeps. The AI agent does the research.


## Current results

The current `train.py` achieves a weighted win_rate of 0.227 after 5 minutes of training on a single H100:

```
random         : 0.950 win_rate (95W 5L 0D / 100)
one_step       : 1.000 win_rate (100W 0L 0D / 100)
minimax_d3     : 0.000 win_rate (0W 100L 0D / 100)
minimax_d5     : 0.000 win_rate (0W 100L 0D / 100)
```

The model beats the random opponent 95% of the time and the one-step-lookahead opponent 100% of the time. It has not yet learned to beat the minimax opponents, which is the current frontier for the autonomous research loop to push against.

The naive baseline (before the fixes described below) achieved only 0.063 win_rate: 82% against random, 0% against everything else, with a training loss that diverged to -800,000.


## What was fixed from the naive baseline

The original train.py used a textbook REINFORCE implementation that failed in practice. Five fixes were required to make training stable and productive:

**Advantage normalization.** Raw advantages (reward minus baseline) were not divided by their standard deviation. As the policy became confident, the gradient magnitudes grew without bound. Dividing by std keeps the gradient scale consistent throughout training.

**Advantage clipping.** Even after normalization, outlier batches (all wins or all losses) produced extreme advantage values. Clamping to the range [-3, 3] adds a safety net.

**Log-probability clipping.** During exploration (15% random moves), the model is forced to take moves it strongly dislikes. Those moves have log probabilities of -10 or -15, which multiply with advantages to produce enormous loss contributions. Clamping log-probs to a minimum of -5.0 prevents any single sample from dominating the gradient.

**Entropy regularization.** Without an entropy bonus, the policy collapses to always playing the same column regardless of board state. Adding `ENTROPY_COEF = 0.03` to the loss rewards the model for maintaining uncertainty across columns, preserving the exploration needed to discover positional play.

**Opponent mixing.** With `SELF_PLAY_RATIO = 1.0`, the model only plays against copies of itself. Two bad players playing each other never learn to block threats or take immediate wins. Changing to 0.7 self-play and 0.3 against the one-step opponent teaches defensive play. This single change took the one-step win rate from 0% to 100%.

The learning rate was also halved from 0.001 to 0.0005 for additional stability. After all fixes, the loss stays bounded between -0.06 and -0.31 for the entire 300-second training run.


## How it works

## Architecture

![Alt text](./architecture-karpathy.png)
![Alt text](./architecture-connect4.png)

The repo has exactly three files that matter, the same three-file architecture as Karpathy's original.

**`prepare.py`** is the immutable harness. In Karpathy's original, this file downloads text data, trains a tokenizer, provides a dataloader, and contains the sacred metric function `evaluate_bpb`. In this Connect Four version, `prepare.py` instead contains the Connect Four game engine (rules, board representation, legal move generation), a fixed suite of four opponents (random player, one-step-lookahead player, minimax depth 3, minimax depth 5), and the sacred metric function `evaluate_winrate` that plays 100 games against each opponent and computes a weighted overall win rate. This file is never modified during experimentation.

**`train.py`** is the single mutable file the AI agent edits. In Karpathy's original, this file contains the GPT model architecture, the Muon+AdamW optimizer, all hyperparameters, and the training loop. In this Connect Four version, `train.py` instead contains a convolutional neural network for evaluating board positions, a REINFORCE policy gradient optimizer, all hyperparameters (network width, exploration rate, self-play ratio, learning rate), and the self-play training loop. Everything in this file is fair game for the agent to change: architecture, optimizer, training procedure, exploration strategy, board encoding, inference-time search. The only constraint is that the code runs without crashing and finishes within the 5-minute time budget.

**`program.md`** is the human's instruction document for the AI agent. It defines the setup procedure, the experiment loop, the logging format, what the agent can and cannot modify, the simplicity criterion, and a list of experiment ideas to try. It also contains the critical "NEVER STOP" directive that keeps the agent running autonomously until the human manually interrupts. This file is written by the human and read by the agent. It is the human's sole lever for steering the autonomous research process.


## The two-level optimization loop

This project has a structural property that Karpathy's original LLM version does not: two nested optimization loops.

The outer loop is the Autoresearch pattern itself. The AI agent edits `train.py`, commits, runs the training script, reads the win_rate from the log, decides whether to keep or discard the change, logs the result, and repeats. This loop runs roughly 12 times per hour, accumulating about 100 experiments overnight.

The inner loop happens inside each 5-minute training run. A neural network plays thousands of Connect Four games against itself and against fixed opponents, wins some, loses some, adjusts its weights based on the outcomes, and gradually develops a strategy from scratch. This inner loop is a standard reinforcement learning process. The neural network starts from random weights each run, learns for 5 minutes playing roughly 39,000 games, and then gets evaluated by `prepare.py`'s `evaluate_winrate`.

The AI agent in the outer loop never plays Connect Four, never watches games, and never learns strategy. It only sees the final win_rate number that comes out of `prepare.py` after each training run. Based on that single number, it decides whether the change it made to `train.py` was an improvement. The outer loop learns which blueprints produce the best inner-loop learners. The inner loop learns how to play Connect Four from that blueprint.


## Project structure

Source files (tracked by git):

```
prepare.py      — game engine, opponents, evaluate_winrate (do not modify)
train.py        — neural network, RL training loop (agent modifies this)
program.md      — agent instructions
analysis.py     — reads results.tsv, generates progress.png chart
report.py       — reads results.tsv + logs/, generates report.md
pyproject.toml  — dependencies
README.md       — this file
```

Generated at runtime (untracked by git):

```
results.tsv     — experiment ledger (one row per experiment)
logs/           — per-experiment full output (named by commit hash)
run.log         — current experiment output (overwritten each cycle)
report.md       — comprehensive morning-after report
progress.png    — visual chart of win_rate over experiments
```


## Quick start on Lightning.ai with an H100

These instructions are written specifically for Lightning.ai Studios, where the terminal cannot open a browser window for OAuth login.


### Step 1: Create a Lightning Studio

Log in to Lightning.ai and create a new Studio with a single H100 GPU. Open the terminal. Lightning Studios use Python 3.12 by default, which is the tested and target version for this project. Verify with:

```bash
python3 --version
```


### Step 2: Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```


### Step 3: Install Claude Code

```bash
npm install -g @anthropic-ai/claude-code
```


### Step 4: Set your Anthropic API key

Claude Code in interactive mode requires OAuth login via a browser, which is not possible inside a Lightning terminal. The workaround is to set your API key as an environment variable. Claude Code's `-p` flag (headless mode) uses this key directly, bypassing OAuth entirely.

Your full API key is approximately 108 characters long and starts with `sk-ant-api03-`. Do not use quotes around the value. Do not truncate it. Copy the entire key from console.anthropic.com and paste it carefully:

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-paste-your-full-key-here
echo 'export ANTHROPIC_API_KEY=sk-ant-api03-paste-your-full-key-here' >> ~/.bashrc
source ~/.bashrc
```

Verify the key is set and has the correct length:

```bash
echo -n "$ANTHROPIC_API_KEY" | wc -c
```

This must print a number between 100 and 120. If it prints 0 or a small number like 20, the key was not pasted completely or the environment variable was not persisted. Re-export the full key and try again.


### Step 5: Clone and set up the repo

```bash
git clone https://github.com/YOUR_USERNAME/autoresearch-connect4.git
cd autoresearch-connect4
uv sync
```


### Step 6: Verify the setup

```bash
uv run prepare.py
```

This runs the self-tests on the game engine and opponents. You should see "All self-tests passed." and a summary of the configuration.


### Step 7: Run a manual test (optional)

```bash
uv run train.py
```

This trains the model for 5 minutes and evaluates it. You should see the loss staying bounded (between -0.05 and -0.35) throughout training, and a final win_rate around 0.227 with 95% against random and 100% against one_step.


### Step 8: Launch the autonomous research loop

This is where the `-p` flag solves the Lightning.ai login problem. Instead of the interactive `claude` command (which requires OAuth via a browser), `-p` sends a prompt directly and uses your API key for authentication. The agent can still edit files, run shell commands, and execute the full experiment loop, it just skips the conversational UI.

Run the entire command on a single line:

```bash
claude -p "Hi, have a look at program.md and let's kick off a new experiment! Let's do the setup first." --dangerously-skip-permissions --max-turns 1000
```

The `--max-turns 1000` flag gives the agent enough headroom to run all night. At roughly 12 experiments per hour, 1000 turns is more than sufficient for 8 hours of autonomous research. The `--dangerously-skip-permissions` flag allows the agent to edit files and run commands without asking for confirmation at each step.

You can now close your laptop, go to sleep, or do anything else. The agent runs autonomously until you manually stop it or it exhausts `--max-turns`.


### Step 9: Review results in the morning

```bash
cd autoresearch-connect4

# Quick summary: see the experiment ledger
cat results.tsv

# Generate a comprehensive markdown report
uv run report.py

# Read the report
cat report.md

# Generate a visual progress chart
uv run analysis.py

# See the git history of kept improvements
git log --oneline

# Review a specific experiment's full training output
ls logs/
cat logs/<commit>.log

# See exactly what train.py looked like at any kept experiment
git show <commit>:train.py

# See what changed between two experiments
git diff <old_commit> <new_commit>
```

The `report.md` file is the most useful starting point. It summarizes the total number of experiments, the keep rate, the improvement timeline showing each kept change and its delta, the biggest single-step wins ranked by impact, any crashes with their error messages, and near-miss discards that came close to improving the win rate and might be worth revisiting. The `logs/` directory preserves the full training output of every experiment, named by commit hash, so you can dig into the per-opponent breakdown for any specific run.


## Troubleshooting


### "Invalid API key · Fix external API key"

This is a known Claude Code bug. The key works for running Python scripts but Claude Code does not read it correctly. Run this diagnostic:

```bash
# Check key length (must be 100-120 characters)
echo -n "$ANTHROPIC_API_KEY" | wc -c

# Check for hidden characters or embedded quotes
echo "$ANTHROPIC_API_KEY" | cat -A

# Clear stale OAuth cache
rm -rf ~/.claude ~/.config/claude-code

# Update Claude Code
npm install -g @anthropic-ai/claude-code@latest

# Test the key directly against the API
curl -s https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-sonnet-4-20250514","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}' \
  | head -c 200
```

If the curl returns a valid JSON response but Claude Code still rejects the key, pass it inline:

```bash
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY claude -p "Hi, have a look at program.md and let's kick off a new experiment! Let's do the setup first." --dangerously-skip-permissions --max-turns 1000
```

Common causes of this error: the key was truncated during paste (check `wc -c` prints 100+), literal quote characters were stored inside the value (check `cat -A` for visible quotes), or a stale OAuth token from a previous session is overriding the environment variable (fix with `rm -rf ~/.claude`).


### Claude prints "y y y y" after launch

This is normal. With `--dangerously-skip-permissions`, Claude Code auto-approves every tool use (file edits, shell commands, git operations) and prints "y" as confirmation. The agent is working. You can close the terminal and come back later.

If you are manually typing "y" each time it pauses, the permissions flag is not taking effect. Make sure the command is on a single line with no backslash line continuations. If that does not help, pre-create the settings file:

```bash
mkdir -p ~/.claude
echo '{"permissions": {"defaultMode": "bypassPermissions"}}' > ~/.claude/settings.json
```

Then rerun the launch command.


## How evaluation works

The `evaluate_winrate` function in `prepare.py` plays the trained model against four opponents of increasing difficulty. Each opponent gets 100 games, alternating who goes first. The model plays greedily during evaluation (argmax of logits, no sampling) so results are deterministic for the three non-random opponents. The overall win_rate is a weighted average:

The random opponent gets a weight of 0.5 (easy to beat, low signal). The one-step-lookahead opponent gets a weight of 1.0 (blocks immediate wins, plays center). The minimax depth-3 opponent gets a weight of 2.0 (looks 3 moves ahead). The minimax depth-5 opponent gets a weight of 3.0 (looks 5 moves ahead, strongest test).

This weighting ensures the metric keeps discriminating as the agent improves. The current model scores 0.227 because it dominates the easy opponents but cannot yet touch the minimax opponents, which carry most of the weight.

The evaluation uses a fixed random seed (42) so that the random opponent plays the same sequence of moves every time, making results reproducible across evaluations of different `train.py` versions.


## The frontier

The autonomous research agent starts from a model that already beats simple opponents. The next breakthroughs should come from changes that teach the model positional play and multi-move planning. Promising directions include adding a value head for better credit assignment, switching from REINFORCE to PPO for more stable policy updates, curriculum learning that gradually introduces minimax opponents into the training mix, and adding Monte Carlo tree search at inference time to extend the model's effective lookahead depth. These are exactly the kinds of experiments the overnight loop is designed to try.


## Design choices

**Same architecture as Karpathy's original.** Three files, same roles, same names. `prepare.py` is the immutable harness with the sacred metric. `train.py` is the single mutable file. `program.md` is the human's instructions. The experiment loop, logging format, simplicity criterion, and git workflow are identical.

**Fixed 5-minute time budget.** Training always runs for exactly 5 minutes of wall-clock time, regardless of what the agent changes. This makes experiments directly comparable whether the agent uses a tiny 2-layer network or a large convolutional model. On an H100, this is enough time for roughly 39,000 games of self-play and 600 training steps.

**Self-contained.** No external dependencies beyond PyTorch. No distributed training, no complex configs. One GPU, one file, one metric.

**Connect Four as the game.** Simple enough that a small neural network can learn meaningful strategy in 5 minutes (thousands of games fit in the budget), complex enough that there are real patterns to discover (center control, double threats, forced wins). The game is solved (first player wins with perfect play), which provides a theoretical ceiling for what the agent could achieve.

**Stable REINFORCE as the baseline.** The baseline includes five fixes that make REINFORCE training reliable (advantage normalization, advantage clipping, log-prob clipping, entropy regularization, opponent mixing). Starting from a working baseline gives the agent a solid foundation to improve from, rather than wasting experiments rediscovering training stability.


## License

MIT
