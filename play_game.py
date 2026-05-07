from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from env import SuperTicTacToeEnv


@dataclass
class GameResult:
    """
    Summary of one completed game.
    """
    winner: int
    is_draw: bool
    move_count: int
    n_turns: int
    history: Optional[List[Dict[str, Any]]] = None


def play_one_game(
    agent_player1,
    agent_player2,
    seed: Optional[int] = None,
    render: bool = False,
    record_history: bool = False,
    max_turns: int = 500,
    illegal_action_mode: str = "raise",
) -> GameResult:
    """
    Play one full game between two agents.

    Player mapping:
      - player 1 uses env.current_player == 1
      - player 2 uses env.current_player == -1

    Args:
        agent_player1: agent for player 1
        agent_player2: agent for player -1
        seed: optional environment seed
        render: whether to call env.render() each turn
        record_history: whether to store step-by-step game history
        max_turns: safety cap to prevent infinite loops
        illegal_action_mode: passed to environment

    Returns:
        GameResult
    """
    env = SuperTicTacToeEnv(seed=seed, illegal_action_mode=illegal_action_mode)
    env.reset(seed=seed)

    history: Optional[List[Dict[str, Any]]] = [] if record_history else None
    n_turns = 0

    if render:
        print("=" * 60)
        print("Initial board")
        env.render()

    while not env.done:
        n_turns += 1
        if n_turns > max_turns:
            raise RuntimeError(
                f"Game exceeded max_turns={max_turns}. "
                "Possible loop or environment bug."
            )

        current_player = env.current_player
        agent = agent_player1 if current_player == 1 else agent_player2

        legal = env.legal_actions()
        action = int(agent.act(env))

        if action not in legal:
            raise RuntimeError(
                f"Agent produced illegal action {action}. "
                f"Current legal actions: {legal[:20]}{'...' if len(legal) > 20 else ''}"
            )

        step_result = env.step(action)

        if record_history:
            history.append(
                {
                    "turn": n_turns,
                    "player": current_player,
                    "agent_name": getattr(agent, "name", agent.__class__.__name__),
                    "intended_action": action,
                    "realized_action": step_result.info.get("realized_action"),
                    "move_forfeited": step_result.info.get("move_forfeited", False),
                    "reward": step_result.reward,
                    "done": step_result.done,
                    "winner": step_result.info.get("winner", 0),
                    "is_draw": step_result.info.get("is_draw", False),
                    "move_count": env.move_count,
                }
            )

        if render:
            print("-" * 60)
            print(
                f"Turn {n_turns} | Player {current_player} | "
                f"Action {action} | Realized {step_result.info.get('realized_action')} | "
                f"Forfeited {step_result.info.get('move_forfeited', False)}"
            )
            env.render()

    return GameResult(
        winner=env.winner,
        is_draw=(env.winner == 0),
        move_count=env.move_count,
        n_turns=n_turns,
        history=history,
    )


if __name__ == "__main__":
    from agents.random_agent import RandomAgent

    p1 = RandomAgent(seed=1, name="Random-P1")
    p2 = RandomAgent(seed=2, name="Random-P2")

    result = play_one_game(
        p1,
        p2,
        seed=42,
        render=True,
        record_history=True,
    )

    print("=" * 60)
    print("Game finished")
    print("Winner:", result.winner)
    print("Is draw:", result.is_draw)
    print("Move count:", result.move_count)
    print("Turns:", result.n_turns)

    if result.history is not None:
        print("Recorded steps:", len(result.history))