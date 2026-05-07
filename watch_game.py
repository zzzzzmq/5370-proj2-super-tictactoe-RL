from __future__ import annotations

import time

from env import SuperTicTacToeEnv
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent


def symbol(player: int) -> str:
    if player == 1:
        return "O"
    if player == -1:
        return "X"
    return "."


def watch_one_game(
    agent1,
    agent2,
    seed: int = 123,
    sleep_time: float = 0.5,
    max_turns: int = 300,
) -> None:
    """
    Watch one full game in the terminal.

    agent1 plays as Player 1, represented by O.
    agent2 plays as Player 2, represented by X.
    """

    env = SuperTicTacToeEnv(seed=seed)
    env.reset()

    print("=" * 70)
    print("Start watching one game")
    print(f"Player 1: {agent1.name} (O)")
    print(f"Player 2: {agent2.name} (X)")
    print("=" * 70)

    env.render()

    turn = 0

    while not env.done and turn < max_turns:
        turn += 1

        current_player = env.current_player
        agent = agent1 if current_player == 1 else agent2

        intended_action = agent.act(env)

        print("\n" + "-" * 70)
        print(f"Turn {turn}")
        print(f"Current player: Player {1 if current_player == 1 else 2} ({symbol(current_player)})")
        print(f"Agent: {agent.name}")
        print(f"Intended action: {intended_action}")

        result = env.step(intended_action)

        obs = result.observation
        reward = result.reward
        done = result.done
        info = result.info

        realized_action = info.get("realized_action", None)
        forfeited = info.get("move_forfeited", None)

        print(f"Realized action: {realized_action}")
        print(f"Move forfeited: {forfeited}")
        print(f"Reward: {reward}")
        print(f"Winner: {env.winner}")
        print(f"Done: {env.done}")
        print(f"Move count: {env.move_count}")

        env.render()

        if sleep_time > 0:
            time.sleep(sleep_time)

    print("\n" + "=" * 70)
    print("Game finished")
    print(f"Total turns: {turn}")
    print(f"Successful move count: {env.move_count}")

    if env.winner == 1:
        print("Winner: Player 1 (O)")
    elif env.winner == -1:
        print("Winner: Player 2 (X)")
    else:
        print("Result: Draw")
    print("=" * 70)


if __name__ == "__main__":
    # Choose matchup here
    agent1 = HeuristicAgent(seed=1)
    agent2 = RandomAgent(seed=2)

    # Other examples:
    # agent1 = RandomAgent(seed=1)
    # agent2 = RandomAgent(seed=2)

    # agent1 = HeuristicAgent(seed=1)
    # agent2 = HeuristicAgent(seed=2)

    watch_one_game(
        agent1=agent1,
        agent2=agent2,
        seed=123,
        sleep_time=0.3,
    )