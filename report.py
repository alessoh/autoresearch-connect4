"""
Autoresearch Connect Four: Morning Report Generator

Reads results.tsv and the logs/ directory to produce a comprehensive
markdown report summarizing the overnight research session.

Usage: uv run report.py
"""

import os
import sys
import glob
from datetime import datetime

import pandas as pd


def load_results():
    """Load results.tsv into a DataFrame."""
    if not os.path.exists("results.tsv"):
        print("No results.tsv found. Run some experiments first.")
        sys.exit(1)
    df = pd.read_csv("results.tsv", sep="\t")
    df["win_rate"] = pd.to_numeric(df["win_rate"], errors="coerce")
    df["memory_gb"] = pd.to_numeric(df["memory_gb"], errors="coerce")
    df["status"] = df["status"].str.strip().str.upper()
    return df


def load_experiment_log(commit):
    """Load the saved log for a specific experiment, if it exists."""
    log_path = os.path.join("logs", f"{commit}.log")
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            return f.read()
    return None


def extract_per_opponent(log_text):
    """Extract per-opponent win rates from a training log."""
    if log_text is None:
        return None
    results = {}
    for line in log_text.split("\n"):
        line = line.strip()
        if "win_rate" in line and "(" in line and "W" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                name = parts[0].strip()
                rest = parts[1].strip()
                try:
                    wr = float(rest.split()[0])
                    results[name] = wr
                except (ValueError, IndexError):
                    pass
    return results if results else None


def generate_report(df):
    """Generate a full markdown report."""
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines.append(f"# Autoresearch Connect Four: Experiment Report")
    lines.append(f"")
    lines.append(f"Generated: {now}")
    lines.append(f"")

    # Overview
    n_total = len(df)
    n_keep = len(df[df["status"] == "KEEP"])
    n_discard = len(df[df["status"] == "DISCARD"])
    n_crash = len(df[df["status"] == "CRASH"])

    lines.append(f"## Overview")
    lines.append(f"")
    lines.append(f"The autonomous research session ran {n_total} experiments. "
                 f"Of these, {n_keep} produced improvements that were kept, "
                 f"{n_discard} were discarded because they did not improve the win rate, "
                 f"and {n_crash} crashed during training or evaluation.")
    if n_keep + n_discard > 0:
        keep_rate = n_keep / (n_keep + n_discard)
        lines.append(f"The keep rate was {keep_rate:.1%}, meaning roughly one in "
                     f"{max(1, round(1/keep_rate)) if keep_rate > 0 else 'infinite'} "
                     f"experiments produced a genuine improvement.")
    lines.append(f"")

    # Baseline and best
    kept = df[df["status"] == "KEEP"].copy()
    if len(kept) > 0:
        baseline_wr = kept.iloc[0]["win_rate"]
        best_wr = kept["win_rate"].max()
        best_row = kept.loc[kept["win_rate"].idxmax()]
        improvement = best_wr - baseline_wr

        lines.append(f"## Key Results")
        lines.append(f"")
        lines.append(f"The baseline model (unmodified train.py) achieved a win rate of "
                     f"{baseline_wr:.4f}. After {n_total} experiments, the best model "
                     f"reached {best_wr:.4f}, an improvement of {improvement:.4f} "
                     f"({improvement / max(baseline_wr, 0.001) * 100:.1f}%). "
                     f"The winning change was: \"{best_row['description']}\".")
        lines.append(f"")

    # Full improvement timeline
    if len(kept) > 1:
        lines.append(f"## Improvement Timeline")
        lines.append(f"")
        lines.append(f"Each row below is a change that improved the win rate and was kept. "
                     f"The improvements are cumulative because each experiment builds on "
                     f"the last kept version of train.py.")
        lines.append(f"")
        lines.append(f"| # | Commit | Win Rate | Delta | Mem (GB) | Description |")
        lines.append(f"|---|--------|----------|-------|----------|-------------|")

        prev_wr = None
        for i, (_, row) in enumerate(kept.iterrows()):
            wr = row["win_rate"]
            delta = f"+{wr - prev_wr:.4f}" if prev_wr is not None else "baseline"
            mem = f"{row['memory_gb']:.1f}" if pd.notna(row["memory_gb"]) else "?"
            commit = str(row["commit"])[:7]
            desc = str(row["description"])
            lines.append(f"| {i+1} | {commit} | {wr:.4f} | {delta} | {mem} | {desc} |")
            prev_wr = wr
        lines.append(f"")

    # Top improvements by delta
    if len(kept) > 2:
        kept_copy = kept.copy()
        kept_copy["prev_wr"] = kept_copy["win_rate"].shift(1)
        kept_copy["delta"] = kept_copy["win_rate"] - kept_copy["prev_wr"]
        hits = kept_copy.iloc[1:].sort_values("delta", ascending=False)

        lines.append(f"## Biggest Wins")
        lines.append(f"")
        lines.append(f"The changes that produced the largest single-step improvements, "
                     f"ranked by delta:")
        lines.append(f"")
        for rank, (_, row) in enumerate(hits.head(5).iterrows(), 1):
            lines.append(f"{rank}. **+{row['delta']:.4f}** (to {row['win_rate']:.4f}): "
                        f"{row['description']}")
        lines.append(f"")

    # Crashes
    crashes = df[df["status"] == "CRASH"]
    if len(crashes) > 0:
        lines.append(f"## Crashes ({len(crashes)} total)")
        lines.append(f"")
        lines.append(f"These experiments failed during training or evaluation. "
                     f"The descriptions indicate what was attempted:")
        lines.append(f"")
        for _, row in crashes.iterrows():
            commit = str(row["commit"])[:7]
            desc = str(row["description"])
            lines.append(f"- {commit}: {desc}")

            log = load_experiment_log(commit)
            if log:
                error_lines = [l.strip() for l in log.split("\n") if "Error" in l or "error" in l]
                if error_lines:
                    lines.append(f"  Error: {error_lines[-1][:100]}")
        lines.append(f"")

    # Notable discards
    discards = df[df["status"] == "DISCARD"]
    if len(discards) > 0:
        lines.append(f"## Notable Discards ({len(discards)} total)")
        lines.append(f"")
        lines.append(f"Changes that were tried but did not improve the win rate. "
                     f"Some of these came close and might be worth revisiting in "
                     f"combination with other changes:")
        lines.append(f"")

        if len(kept) > 0:
            best_so_far = kept["win_rate"].max()
            near_misses = discards[discards["win_rate"] > 0].copy()
            if len(near_misses) > 0:
                near_misses["gap"] = best_so_far - near_misses["win_rate"]
                near_misses = near_misses.sort_values("gap").head(5)
                for _, row in near_misses.iterrows():
                    desc = str(row["description"])
                    wr = row["win_rate"]
                    lines.append(f"- {desc} (win_rate {wr:.4f}, gap of {row['gap']:.4f})")
        lines.append(f"")

    # Experiment logs available
    log_files = glob.glob("logs/*.log")
    if log_files:
        lines.append(f"## Experiment Logs")
        lines.append(f"")
        lines.append(f"Detailed logs for {len(log_files)} experiments are saved in "
                     f"the logs/ directory. Each file is named by its git commit hash. "
                     f"To review a specific experiment's full output:")
        lines.append(f"")
        lines.append(f"```")
        lines.append(f"cat logs/<commit>.log")
        lines.append(f"```")
        lines.append(f"")

    # Git instructions
    lines.append(f"## Reviewing the Code")
    lines.append(f"")
    lines.append(f"To see the exact train.py for any kept experiment, check out "
                 f"that commit:")
    lines.append(f"")
    lines.append(f"```")
    lines.append(f"git log --oneline          # see the full history")
    lines.append(f"git show <commit>:train.py  # see train.py at that point")
    lines.append(f"git diff <old> <new>        # see what changed between two experiments")
    lines.append(f"```")
    lines.append(f"")
    lines.append(f"To generate the progress chart:")
    lines.append(f"")
    lines.append(f"```")
    lines.append(f"uv run analysis.py")
    lines.append(f"```")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_results()

    print(f"Loaded {len(df)} experiments from results.tsv")
    print()

    report = generate_report(df)

    report_path = "report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report written to {report_path}")
    print()

    # Also print a quick summary to terminal
    kept = df[df["status"] == "KEEP"]
    if len(kept) > 0:
        baseline = kept.iloc[0]["win_rate"]
        best = kept["win_rate"].max()
        print(f"Baseline win_rate: {baseline:.4f}")
        print(f"Best win_rate:     {best:.4f}")
        print(f"Improvement:       +{best - baseline:.4f}")
    print(f"Experiments:       {len(df)} total, {len(kept)} kept, "
          f"{len(df[df['status'] == 'CRASH'])} crashed")
