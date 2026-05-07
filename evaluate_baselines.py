from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np

from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from play_game import GameResult, play_one_game


@dataclass
class MatchupStats:
    """
    Aggregate statistics over many games.
    """
    matchup_name: str
    n_games: int

    player1_wins: int
    player2_wins: int
    draws: int

    player1_win_rate: float
    player2_win_rate: float
    draw_rate: float

    avg_move_count: float
    avg_turns: float

    def to_dict(self) -> Dict:
        return asdict(self)


def evaluate_matchup(
    agent_player1,
    agent_player2,
    n_games: int = 100,
    base_seed: int = 123,
    render_first_n: int = 0,
    record_first_n: int = 0,
) -> MatchupStats:
    """
    Evaluate two agents over multiple games.

    Args:
        agent_player1: agent for player 1
        agent_player2: agent for player -1
        n_games: number of games
        base_seed: base seed for reproducibility
        render_first_n: render the first few games for inspection
        record_first_n: record detailed histories for first few games

    Returns:
        MatchupStats
    """
    p1_wins = 0
    p2_wins = 0
    draws = 0

    move_counts: List[int] = []
    turns_list: List[int] = []

    for g in range(n_games):
        seed = base_seed + g

        result: GameResult = play_one_game(
            agent_player1=agent_player1,
            agent_player2=agent_player2,
            seed=seed,
            render=(g < render_first_n),
            record_history=(g < record_first_n),
        )

        if result.winner == 1:
            p1_wins += 1
        elif result.winner == -1:
            p2_wins += 1
        else:
            draws += 1

        move_counts.append(result.move_count)
        turns_list.append(result.n_turns)

    matchup_name = (
        f"{getattr(agent_player1, 'name', agent_player1.__class__.__name__)} "
        f"vs "
        f"{getattr(agent_player2, 'name', agent_player2.__class__.__name__)}"
    )

    return MatchupStats(
        matchup_name=matchup_name,
        n_games=n_games,
        player1_wins=p1_wins,
        player2_wins=p2_wins,
        draws=draws,
        player1_win_rate=p1_wins / n_games,
        player2_win_rate=p2_wins / n_games,
        draw_rate=draws / n_games,
        avg_move_count=float(np.mean(move_counts)),
        avg_turns=float(np.mean(turns_list)),
    )


def print_matchup_stats(stats: MatchupStats) -> None:
    """
    Nicely print matchup statistics.
    """
    print("=" * 70)
    print(f"Matchup: {stats.matchup_name}")
    print(f"Games: {stats.n_games}")
    print("-" * 70)
    print(f"Player 1 wins : {stats.player1_wins:4d}  ({stats.player1_win_rate:.3f})")
    print(f"Player 2 wins : {stats.player2_wins:4d}  ({stats.player2_win_rate:.3f})")
    print(f"Draws         : {stats.draws:4d}  ({stats.draw_rate:.3f})")
    print(f"Avg move count: {stats.avg_move_count:.2f}")
    print(f"Avg turns     : {stats.avg_turns:.2f}")
    print("=" * 70)


def run_random_vs_random(
    n_games: int = 200,
    base_seed: int = 123,
) -> MatchupStats:
    """
    Convenience wrapper for Random vs Random baseline.
    """
    p1 = RandomAgent(seed=base_seed + 1000, name="RandomAgent-P1")
    p2 = RandomAgent(seed=base_seed + 2000, name="RandomAgent-P2")

    stats = evaluate_matchup(
        agent_player1=p1,
        agent_player2=p2,
        n_games=n_games,
        base_seed=base_seed,
        render_first_n=0,
        record_first_n=0,
    )
    return stats


def run_heuristic_vs_random(
    n_games: int = 200,
    base_seed: int = 123,
) -> MatchupStats:
    """
    Convenience wrapper for HeuristicAgent vs RandomAgent.
    """
    p1 = HeuristicAgent(seed=base_seed + 1000, name="HeuristicAgent-P1")
    p2 = RandomAgent(seed=base_seed + 2000, name="RandomAgent-P2")

    stats = evaluate_matchup(
        agent_player1=p1,
        agent_player2=p2,
        n_games=n_games,
        base_seed=base_seed,
        render_first_n=0,
        record_first_n=0,
    )
    return stats


def run_heuristic_vs_heuristic(
    n_games: int = 200,
    base_seed: int = 123,
) -> MatchupStats:
    """
    Convenience wrapper for HeuristicAgent vs HeuristicAgent.
    """
    p1 = HeuristicAgent(seed=base_seed + 1000, name="HeuristicAgent-P1")
    p2 = HeuristicAgent(seed=base_seed + 2000, name="HeuristicAgent-P2")

    stats = evaluate_matchup(
        agent_player1=p1,
        agent_player2=p2,
        n_games=n_games,
        base_seed=base_seed,
        render_first_n=0,
        record_first_n=0,
    )
    return stats


if __name__ == "__main__":
    # 运行 RandomAgent vs RandomAgent
    print("Running Random vs Random baseline:")
    stats = run_random_vs_random(n_games=200, base_seed=123)
    print_matchup_stats(stats)

    # 运行 HeuristicAgent vs RandomAgent
    print("\nRunning HeuristicAgent vs RandomAgent:")
    stats = run_heuristic_vs_random(n_games=200, base_seed=123)
    print_matchup_stats(stats)

    # 运行 HeuristicAgent vs HeuristicAgent
    print("\nRunning HeuristicAgent vs HeuristicAgent:")
    stats = run_heuristic_vs_heuristic(n_games=200, base_seed=123)
    print_matchup_stats(stats)