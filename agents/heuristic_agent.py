from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import numpy as np

from env import SuperTicTacToeEnv


@dataclass
class HeuristicAgent:
    """
    A simple one-step heuristic baseline agent.

    Strategy:
    1. If the current player has an immediate winning move, take it.
    2. Otherwise, if the opponent has an immediate winning move, block it.
    3. Otherwise, choose a random legal action.

    Important:
    - This heuristic uses a deterministic proxy: it assumes the intended action
      is placed exactly at the selected square when checking immediate wins.
    - The actual environment is stochastic, so this is not an optimal policy.
    - It is still useful as a strong baseline against RandomAgent.
    """

    seed: Optional[int] = None
    name: str = "HeuristicAgent"

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def act(self, env: SuperTicTacToeEnv) -> int:
        """
        Select an action according to the heuristic rule.

        Args:
            env: current SuperTicTacToeEnv instance.

        Returns:
            A legal intended action.
        """
        legal_actions = env.legal_actions()

        if len(legal_actions) == 0:
            raise RuntimeError(f"{self.name} found no legal actions to play.")

        current_player = env.current_player
        opponent = -current_player

        # 1. If current player can win immediately, take the winning move.
        winning_actions = self._find_winning_actions(
            env=env,
            player=current_player,
            legal_actions=legal_actions,
        )
        if winning_actions:
            return int(self.rng.choice(winning_actions))

        # 2. If opponent can win immediately, block one of those moves.
        blocking_actions = self._find_winning_actions(
            env=env,
            player=opponent,
            legal_actions=legal_actions,
        )
        if blocking_actions:
            return int(self.rng.choice(blocking_actions))

        # 3. Otherwise, play randomly.
        return int(self.rng.choice(legal_actions))

    def _find_winning_actions(
        self,
        env: SuperTicTacToeEnv,
        player: int,
        legal_actions: List[int],
    ) -> List[int]:
        """
        Return all legal actions that would complete a winning line for `player`
        under a deterministic-placement approximation.

        Args:
            env: current environment.
            player: player to test, either 1 or -1.
            legal_actions: currently legal intended actions.

        Returns:
            List of actions that would immediately win for `player`.
        """
        winning_actions: List[int] = []

        for action in legal_actions:
            if self._would_win_deterministically(env, player, action):
                winning_actions.append(action)

        return winning_actions

    def _would_win_deterministically(
        self,
        env: SuperTicTacToeEnv,
        player: int,
        action: int,
    ) -> bool:
        """
        Check whether placing `player` exactly at `action` would produce a win.

        This function temporarily modifies env.board and then restores it.

        Args:
            env: current environment.
            player: player to test.
            action: candidate action.

        Returns:
            True if exact placement at action wins for player.
        """
        if env.board[action] != 0:
            return False

        old_value = env.board[action]
        env.board[action] = player

        winner = env.check_winner()

        env.board[action] = old_value

        return winner == player