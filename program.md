# autoresearch-connect4

This is an experiment to have an LLM do its own research on game-playing agent architecture.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr3`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, game engine, opponents, evaluation. Do not modify.
   - `train.py` — the file you modify. Neural network architecture, RL training loop, hyperparameters.
4. **Verify setup**: Run `uv run prepare.py` to confirm the game engine and opponents work.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: neural network architecture, optimizer, hyperparameters, training loop, exploration strategy, self-play ratio, board encoding, inference-time search (MCTS), batch size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the game engine, fixed opponents, and the `evaluate_winrate` function.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_winrate` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the highest win_rate.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the training procedure, the exploration strategy. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful win_rate gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.005 win_rate improvement that adds 30 lines of hacky code? Probably not worth it. A 0.005 win_rate improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
win_rate:          0.312000
training_seconds:  300.1
total_seconds:     325.9
eval_seconds:      25.8
peak_vram_mb:      1024.0
total_games:       12800
num_steps:         200
num_params:        125000
conv_layers:       3
conv_channels:     64
fc_hidden:         128
```

You can extract the key metric from the log file:

```
grep "^win_rate:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	win_rate	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. win_rate achieved (e.g. 0.312000) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 1.0 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	win_rate	memory_gb	status	description
a1b2c3d	0.312000	1.0	keep	baseline
b2c3d4e	0.345000	1.0	keep	increase conv channels to 128
c3d4e5f	0.298000	1.0	discard	switch to tanh activation
d4e5f6g	0.000000	0.0	crash	double model size (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr3`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^win_rate:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Save the experiment log: `mkdir -p logs && cp run.log logs/$(git rev-parse --short HEAD).log` — this preserves the full output of every experiment so the human can review them later.
8. Record the results in the tsv (NOTE: do not commit the results.tsv file or the logs directory, leave them untracked by git)
9. If win_rate improved (higher), you "advance" the branch, keeping the git commit
10. If win_rate is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for evaluation overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

## Experiment ideas to try

Here are some directions to explore, roughly ordered from simple to radical:

**Architecture changes:**
- Vary conv channels (32, 64, 128, 256)
- Vary number of conv layers (2, 3, 4, 5)
- Vary FC hidden size (64, 128, 256, 512)
- Add residual connections between conv layers
- Try different activation functions (LeakyReLU, GELU, SiLU)
- Add a value head alongside the policy head
- Try separating policy and value into distinct tower branches

**Training procedure changes:**
- Vary learning rate (0.0001, 0.001, 0.01)
- Vary exploration rate (0.05, 0.10, 0.15, 0.25)
- Change self-play ratio (try mixing in opponent games)
- Add a replay buffer to reuse past experience
- Try PPO or A2C instead of REINFORCE
- Add entropy bonus to encourage exploration
- Try curriculum learning (train vs random first, then harder opponents)
- Add learning rate warmup or cosine schedule

**Board encoding changes:**
- Add a third channel encoding whose turn it is
- Add channels for threat detection (three-in-a-row positions)
- Try a flat encoding instead of convolutional

**Inference-time search:**
- Add Monte Carlo tree search at evaluation time
- Try one-step lookahead during evaluation
