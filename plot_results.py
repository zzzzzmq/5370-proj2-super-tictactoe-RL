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
    D阶段：DQN训练胜率曲线
    """
    ensure_dir(save_dir)

    episodes = [
        100, 200, 300, 400, 500,
        600, 700, 800, 900, 1000,
        1100, 1200, 1300, 1400, 1500,
        1600, 1700, 1800, 1900, 2000,
    ]

    win_rate = [
        0.720, 0.700, 0.580, 0.520, 0.720,
        0.600, 0.620, 0.500, 0.460, 0.500,
        0.600, 0.740, 0.440, 0.500, 0.640,
        0.580, 0.520, 0.480, 0.460, 0.600,
    ]

    loss_rate = [
        0.280, 0.300, 0.420, 0.480, 0.280,
        0.400, 0.380, 0.500, 0.540, 0.500,
        0.400, 0.260, 0.560, 0.500, 0.360,
        0.420, 0.480, 0.520, 0.540, 0.400,
    ]

    avg_return = [
        0.440, 0.400, 0.160, 0.040, 0.440,
        0.200, 0.240, 0.000, -0.080, 0.000,
        0.200, 0.480, -0.120, 0.000, 0.280,
        0.160, 0.040, -0.040, -0.080, 0.200,
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, win_rate, marker="o", label="Win Rate")
    plt.plot(episodes, loss_rate, marker="s", label="Loss Rate")
    plt.plot(episodes, avg_return, marker="^", label="Avg Return")

    baseline_random_p1 = 0.545
    plt.axhline(
        y=baseline_random_p1,
        linestyle="--",
        label="Random-vs-Random P1 Baseline",
    )

    plt.ylim(-0.15, 1.05)
    plt.xlabel("Training Episode")
    plt.ylabel("Value")
    plt.title("DQN Training Performance vs Random Opponent")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "dqn_training_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_dqn_vs_baselines(save_dir: str = "figures") -> None:
    """
    D阶段：DQN与baseline对比图
    """
    ensure_dir(save_dir)

    labels = [
        "Random-vs-Random\n(P1 win rate)",
        "Heuristic-vs-Random\n(P1 win rate)",
        "DQN Best Eval\n(ep 1200)",
        "DQN Last Eval\n(ep 2000)",
    ]

    values = [0.545, 0.985, 0.740, 0.600]

    x = list(range(len(labels)))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, values)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.xticks(x, labels, rotation=10)
    plt.ylim(0, 1.05)
    plt.ylabel("Win Rate")
    plt.title("Comparison Between Baselines and DQN Results")
    plt.tight_layout()

    save_path = os.path.join(save_dir, "dqn_vs_baselines.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main() -> None:
    plot_baseline_winrates()
    plot_baseline_lengths()
    plot_dqn_training_curve()
    plot_dqn_vs_baselines()
    print("All figures generated successfully.")


if __name__ == "__main__":
    main()