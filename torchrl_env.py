from __future__ import annotations

from typing import Optional, Dict, Any, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env import SuperTicTacToeEnv
from agents.random_agent import RandomAgent


class SingleAgentVsRandomEnv(gym.Env):
    """
    Gym-style single-agent wrapper for SuperTicTacToeEnv.

    Design:
    - RL agent always plays as Player 1
    - Opponent is a fixed RandomAgent
    - One external step(action) does:
        1) RL agent takes one move
        2) if game not finished, RandomAgent takes one move
        3) return the next state back to RL agent

    Observation:
    - same as base env: shape (97,)
      first 96 dims = board
      last dim = current_player

    Action:
    - Discrete(96), intended square index

    Info:
    - includes action_mask for legal intended actions
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        seed: Optional[int] = None,
        opponent_seed: Optional[int] = None,
        illegal_action_mode: str = "raise",
    ) -> None:
        super().__init__()

        self.base_seed = seed
        self.env = SuperTicTacToeEnv(
            seed=seed,
            illegal_action_mode=illegal_action_mode,
        )
        self.opponent = RandomAgent(
            seed=opponent_seed if opponent_seed is not None else seed,
            name="RandomOpponent",
        )

        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(97,),
            dtype=np.int8,
        )
        self.action_space = spaces.Discrete(96)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.base_seed = seed

        obs = self.env.reset(seed=seed)

        # RL agent should always act as player 1 at the start of wrapper step
        if self.env.current_player != 1:
            raise RuntimeError("Wrapper reset invariant violated: current_player should be 1.")

        info = self._build_info(stage="reset", extra={})
        return obs, info

    def step(self, action: int):
        if self.env.done:
            raise RuntimeError("Environment is done. Please call reset() first.")

        if self.env.current_player != 1:
            raise RuntimeError(
                f"Wrapper invariant violated: expected player 1 to act, got {self.env.current_player}."
            )

        # -------------------------------------------------
        # 1) RL agent acts
        # -------------------------------------------------
        agent_result = self.env.step(int(action))

        # If game ends right after RL agent move
        if agent_result.done:
            reward = self._reward_from_env_winner(self.env.winner)
            obs = agent_result.observation
            info = self._build_info(
                stage="agent_terminal",
                extra={
                    "winner": self.env.winner,
                    "agent_action": int(action),
                    "agent_realized_action": agent_result.info.get("realized_action"),
                    "agent_move_forfeited": agent_result.info.get("move_forfeited", False),
                },
            )
            terminated = True
            truncated = False
            return obs, reward, terminated, truncated, info

        # -------------------------------------------------
        # 2) Opponent acts automatically
        # -------------------------------------------------
        if self.env.current_player != -1:
            raise RuntimeError(
                f"Expected opponent turn after agent move, got {self.env.current_player}."
            )

        opponent_action = self.opponent.act(self.env)
        opp_result = self.env.step(int(opponent_action))

        # If game ends after opponent move
        if opp_result.done:
            reward = self._reward_from_env_winner(self.env.winner)
            obs = opp_result.observation
            info = self._build_info(
                stage="opponent_terminal",
                extra={
                    "winner": self.env.winner,
                    "agent_action": int(action),
                    "agent_realized_action": agent_result.info.get("realized_action"),
                    "agent_move_forfeited": agent_result.info.get("move_forfeited", False),
                    "opponent_action": int(opponent_action),
                    "opponent_realized_action": opp_result.info.get("realized_action"),
                    "opponent_move_forfeited": opp_result.info.get("move_forfeited", False),
                },
            )
            terminated = True
            truncated = False
            return obs, reward, terminated, truncated, info

        # -------------------------------------------------
        # 3) Non-terminal transition, should be back to player 1
        # -------------------------------------------------
        if self.env.current_player != 1:
            raise RuntimeError(
                f"Expected player 1 turn after opponent move, got {self.env.current_player}."
            )

        obs = self.env.get_observation()
        reward = 0.0
        info = self._build_info(
            stage="midgame",
            extra={
                "winner": 0,
                "agent_action": int(action),
                "agent_realized_action": agent_result.info.get("realized_action"),
                "agent_move_forfeited": agent_result.info.get("move_forfeited", False),
                "opponent_action": int(opponent_action),
                "opponent_realized_action": opp_result.info.get("realized_action"),
                "opponent_move_forfeited": opp_result.info.get("move_forfeited", False),
            },
        )
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, info

    def render(self):
        self.env.render()

    # -------------------------------------------------
    # helpers
    # -------------------------------------------------
    def _reward_from_env_winner(self, winner: int) -> float:
        """
        Reward from RL agent (player 1) perspective.
        """
        if winner == 1:
            return 1.0
        if winner == -1:
            return -1.0
        return 0.0

    def _build_info(self, stage: str, extra: Dict[str, Any]) -> Dict[str, Any]:
        info = {
            "stage": stage,
            "current_player": self.env.current_player,
            "winner": self.env.winner,
            "done": self.env.done,
            "move_count": self.env.move_count,
            "action_mask": self.env.get_action_mask().astype(np.int8),
        }
        info.update(extra)
        return info