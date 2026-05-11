from __future__ import annotations

import os
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def plot_baseline_winrates(save_dir: str = "figures") -> None:
    """
    Phase C: baseline win-rate comparison.
    Uses the latest results after fixing HeuristicAgent's current_player logic.
    """
    ensure_dir(save_dir)

    matchups = [
        "Random vs Random",
        "Heuristic vs Random",
        "Heuristic vs Heuristic",
    ]

    p1_win = [0.545, 0.985, 0.515]
    p2_win = [0.455, 0.015, 0.485]
    draw = [0.000, 0.000, 0.000]

    x = list(range(len(matchups)))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(
        [i - width for i in x],
        p1_win,
        width=width,
        label="Player 1 Win Rate",
    )
    bars2 = ax.bar(
        x,
        p2_win,
        width=width,
        label="Player 2 Win Rate",
    )
    bars3 = ax.bar(
        [i + width for i in x],
        draw,
        width=width,
        label="Draw Rate",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(matchups, rotation=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Rate")
    ax.set_title("Baseline Matchup Win Rates")
    ax.legend()

    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(bars3, fmt="%.3f", padding=3, fontsize=9)

    fig.tight_layout()

    save_path = os.path.join(save_dir, "baseline_winrate.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_baseline_lengths(save_dir: str = "figures") -> None:
    """
    Phase C: baseline average game length comparison.
    Uses the latest results after fixing HeuristicAgent's current_player logic.
    """
    ensure_dir(save_dir)

    matchups = [
        "Random vs Random",
        "Heuristic vs Random",
        "Heuristic vs Heuristic",
    ]

    avg_move_count = [48.85, 32.20, 33.67]
    avg_turns = [64.80, 40.80, 43.28]

    x = list(range(len(matchups)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(
        [i - width / 2 for i in x],
        avg_move_count,
        width=width,
        label="Avg Move Count",
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x],
        avg_turns,
        width=width,
        label="Avg Turns",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(matchups, rotation=10)
    ax.set_ylabel("Average Length")
    ax.set_title("Baseline Matchup Average Game Length")
    ax.legend()

    ax.bar_label(bars1, fmt="%.2f", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.2f", padding=3, fontsize=9)

    fig.tight_layout()

    save_path = os.path.join(save_dir, "baseline_lengths.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_dqn_training_curve(save_dir: str = "figures") -> None:
    """
    Phase D: DQN training curve against RandomAgent.
    """
    ensure_dir(save_dir)

    episodes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    win_rate = [0.62, 0.66, 0.60, 0.66, 0.58, 0.60, 0.54, 0.70, 0.64, 0.54]
    loss_rate = [0.38, 0.34, 0.40, 0.34, 0.42, 0.40, 0.46, 0.30, 0.36, 0.46]
    avg_return = [0.24, 0.32, 0.20, 0.32, 0.16, 0.20, 0.08, 0.40, 0.28, 0.08]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(episodes, win_rate, marker="o", label="Win Rate")
    ax.plot(episodes, loss_rate, marker="s", label="Loss Rate")
    ax.plot(episodes, avg_return, marker="^", label="Avg Return")

    baseline_random_p1 = 0.545
    ax.axhline(
        y=baseline_random_p1,
        linestyle="--",
        label="Random-vs-Random P1 Baseline",
    )

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Value")
    ax.set_title("DQN Training Performance vs Random Opponent")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    save_path = os.path.join(save_dir, "dqn_training_curve.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_dqn_vs_baselines(save_dir: str = "figures") -> None:
    """
    Phase D: comparison between baseline strategies and DQN results.
    """
    ensure_dir(save_dir)

    labels = [
        "Random-vs-Random\n(P1 win rate)",
        "Heuristic-vs-Random\n(P1 win rate)",
        "DQN Best Eval\n(win rate)",
        "DQN Final Eval\n(win rate)",
    ]

    values = [0.545, 0.960, 0.700, 0.540]

    x = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(x, values)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Win Rate")
    ax.set_title("Comparison Between Baselines and DQN Results")

    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)

    fig.tight_layout()

    save_path = os.path.join(save_dir, "dqn_vs_baselines.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main() -> None:
    plot_baseline_winrates()
    plot_baseline_lengths()
    plot_dqn_training_curve()
    plot_dqn_vs_baselines()
    print("All figures generated successfully.")


if __name__ == "__main__":
    main()