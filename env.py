from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from geometry import GeometryData, build_geometry_data


Player = int
Action = int
Observation = np.ndarray


@dataclass
class StepResult:
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]


class SuperTicTacToeEnv:
    """
    Stochastic Super Tic-Tac-Toe environment.

    Current conventions:
    - board is a flat array of length 96
    - board[i] in {0, 1, -1}
    - current_player in {1, -1}
    - action is an intended square index in [0, 95]
    - actual placement is stochastic:
        * chosen square with prob 1/2
        * each of 8 same-level neighbors with prob 1/16
    - if sampled square is out of board or occupied, move is forfeited
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        illegal_action_mode: str = "raise",
    ) -> None:
        """
        Args:
            seed:
                Random seed for reproducibility.
            illegal_action_mode:
                How to handle illegal chosen actions (occupied square, out of range).
                Options:
                    - "raise": raise ValueError
                    - "forfeit": treat as forfeited move
        """
        if illegal_action_mode not in {"raise", "forfeit"}:
            raise ValueError("illegal_action_mode must be 'raise' or 'forfeit'")

        self.geometry: GeometryData = build_geometry_data()
        self.rng = np.random.default_rng(seed)
        self.illegal_action_mode = illegal_action_mode

        # Core environment state
        self.board: np.ndarray = np.zeros(96, dtype=np.int8)
        self.current_player: Player = 1
        self.done: bool = False
        self.winner: int = 0  # 0 means no winner yet; 1 or -1 after terminal win
        self.move_count: int = 0

        # Optional bookkeeping
        self.last_action: Optional[int] = None          # intended action
        self.last_realized_action: Optional[int] = None # actual realized placement index
        self.last_move_forfeited: bool = False

    # =========================
    # Core API
    # =========================
    def reset(self, seed: Optional[int] = None) -> Observation:
        """
        Reset environment state and return initial observation.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.board = np.zeros(96, dtype=np.int8)
        self.current_player = 1
        self.done = False
        self.winner = 0
        self.move_count = 0

        self.last_action = None
        self.last_realized_action = None
        self.last_move_forfeited = False

        return self.get_observation()

    def step(self, action: Action) -> StepResult:
        """
        Execute one environment step.

        Flow:
        1. validate / handle intended action
        2. resolve stochastic realized location
        3. apply move if valid realized location exists
        4. check winner
        5. check draw
        6. switch player if not terminal

        Returns:
            StepResult(observation, reward, done, info)
        """
        if self.done:
            raise RuntimeError("Cannot call step() after environment is done. Please reset().")

        self.last_action = action
        self.last_realized_action = None
        self.last_move_forfeited = False

        info: Dict[str, Any] = {
            "current_player_before_step": self.current_player,
            "intended_action": action,
            "realized_action": None,
            "move_forfeited": False,
            "winner": 0,
            "is_draw": False,
        }

        # --------
        # 1. Handle illegal intended action
        # --------
        intended_legal = self.is_legal_action(action)

        if not intended_legal:
            if self.illegal_action_mode == "raise":
                raise ValueError(f"Illegal action: {action}")
            elif self.illegal_action_mode == "forfeit":
                self.last_move_forfeited = True
                info["move_forfeited"] = True

                # No board change; turn still ends
                reward = self._reward_for_nonterminal_forfeit()
                self._switch_player()

                obs = self.get_observation()
                return StepResult(obs, reward, self.done, info)

        # --------
        # 2. Resolve stochastic realized action
        # --------
        realized_action = self.resolve_action(action)
        self.last_realized_action = realized_action
        info["realized_action"] = realized_action

        # --------
        # 3. Apply move if valid realized location exists
        # --------
        if realized_action is None:
            # forfeited due to stochastic outcome landing outside / occupied
            self.last_move_forfeited = True
            info["move_forfeited"] = True

            reward = self._reward_for_nonterminal_forfeit()

            # No board update, but turn ends
            self._switch_player()

            obs = self.get_observation()
            return StepResult(obs, reward, self.done, info)

        # successful realized placement
        self.board[realized_action] = self.current_player
        self.move_count += 1

        # --------
        # 4. Check winner
        # --------
        winner = self.check_winner()
        if winner != 0:
            self.done = True
            self.winner = winner
            info["winner"] = winner

            reward = self._reward_for_terminal_win(winner)
            obs = self.get_observation()
            return StepResult(obs, reward, self.done, info)

        # --------
        # 5. Check draw
        # --------
        if self.is_draw():
            self.done = True
            self.winner = 0
            info["is_draw"] = True

            reward = self._reward_for_draw()
            obs = self.get_observation()
            return StepResult(obs, reward, self.done, info)

        # --------
        # 6. Switch player and continue
        # --------
        reward = self._reward_for_nonterminal_step()
        self._switch_player()

        obs = self.get_observation()
        return StepResult(obs, reward, self.done, info)

    # =========================
    # Observation / action mask
    # =========================
    def get_observation(self) -> Observation:
        """
        Return current observation.

        Current simple encoding:
        - 96 board entries in {-1, 0, 1}
        - 1 extra entry = current_player

        Shape: (97,)
        """
        obs = np.zeros(97, dtype=np.int8)
        obs[:96] = self.board
        obs[96] = self.current_player
        return obs

    def get_action_mask(self) -> np.ndarray:
        """
        Boolean mask of shape (96,), True for legal intended actions.
        """
        return self.board == 0

    def legal_actions(self) -> List[int]:
        """
        Return all currently legal intended actions.
        """
        return np.flatnonzero(self.board == 0).tolist()

    # =========================
    # Action legality
    # =========================
    def is_legal_action(self, action: int) -> bool:
        """
        Intended action must:
        - be an integer index in [0, 95]
        - point to an empty square
        """
        if not isinstance(action, (int, np.integer)):
            return False
        if not (0 <= int(action) < 96):
            return False
        if self.board[int(action)] != 0:
            return False
        return True

    # =========================
    # Stochastic move resolution
    # =========================
    def resolve_action(self, action: int) -> Optional[int]:
        """
        Resolve intended action into an actual realized square.

        Probabilities:
        - intended square with probability 1/2
        - each of 8 neighbors with probability 1/16

        If sampled square is invalid (outside board) or occupied,
        return None (forfeited move).
        """
        action = int(action)

        # Candidate list:
        # slot 0 -> intended square
        # slots 1..8 -> 8 neighbors in fixed direction order
        candidates: List[Optional[int]] = [action] + self.geometry.neighbors[action]

        # probability vector
        probs = np.array([0.5] + [1.0 / 16.0] * 8, dtype=float)

        sampled_slot = int(self.rng.choice(len(candidates), p=probs))
        sampled_index = candidates[sampled_slot]

        # outside board
        if sampled_index is None:
            return None

        # occupied
        if self.board[sampled_index] != 0:
            return None

        return sampled_index

    # =========================
    # Winner / terminal checks
    # =========================
    def check_winner(self) -> int:
        """
        Check whether any player occupies a full winning line.

        Returns:
            1  if player 1 wins
            -1 if player -1 wins
            0  otherwise
        """
        for line in self.geometry.winning_lines:
            values = self.board[line]
            s = int(values.sum())

            if s == len(line):
                return 1
            if s == -len(line):
                return -1

        return 0

    def is_draw(self) -> bool:
        """
        Draw iff board is full and no winner exists.
        """
        if np.any(self.board == 0):
            return False
        return self.check_winner() == 0

    # =========================
    # Rewards
    # =========================
    def _reward_for_terminal_win(self, winner: int) -> float:
        """
        Reward from the perspective of the player who just acted.

        Since step() is called by the current_player before switching,
        if winner == current_player, reward should be +1.
        """
        return 1.0

    def _reward_for_draw(self) -> float:
        return 0.0

    def _reward_for_nonterminal_step(self) -> float:
        return 0.0

    def _reward_for_nonterminal_forfeit(self) -> float:
        return 0.0

    # =========================
    # State helpers
    # =========================
    def _switch_player(self) -> None:
        self.current_player *= -1

    def clone(self) -> "SuperTicTacToeEnv":
        """
        Optional utility for debugging / search / testing.
        """
        env = SuperTicTacToeEnv(illegal_action_mode=self.illegal_action_mode)
        env.geometry = self.geometry
        env.rng = np.random.default_rng()

        env.board = self.board.copy()
        env.current_player = self.current_player
        env.done = self.done
        env.winner = self.winner
        env.move_count = self.move_count

        env.last_action = self.last_action
        env.last_realized_action = self.last_realized_action
        env.last_move_forfeited = self.last_move_forfeited
        return env

    # =========================
    # Rendering
    # =========================
    def render(self) -> None:
        """
        Simple text render by level.
        """
        symbol_map = {
            0: ".",
            1: "O",
            -1: "X",
        }

        print("=" * 50)
        print(f"Current player: {'O (1)' if self.current_player == 1 else 'X (-1)'}")
        print(f"Done: {self.done}, Winner: {self.winner}, Move count: {self.move_count}")
        print(f"Last intended action: {self.last_action}")
        print(f"Last realized action: {self.last_realized_action}")
        print(f"Last move forfeited: {self.last_move_forfeited}")
        print("-" * 50)

        for level in (1, 2, 3):
            print(f"Level {level}")
            width = self._level_width(level)

            grid = [[" " for _ in range(width)] for _ in range(4)]

            for idx, (lv, gr, gc) in self.geometry.index_to_geom.items():
                if lv == level:
                    grid[gr][gc] = symbol_map[int(self.board[idx])]

            for row in grid:
                print(" ".join(row))
            print()

    def render_indices(self) -> None:
        """
        Debug render showing index layout instead of symbols.
        """
        print("=" * 50)
        print("Index layout")

        for level in (1, 2, 3):
            print(f"Level {level}")
            width = self._level_width(level)
            grid = [["   ." for _ in range(width)] for _ in range(4)]

            for idx, (lv, gr, gc) in self.geometry.index_to_geom.items():
                if lv == level:
                    grid[gr][gc] = f"{idx:4d}"

            for row in grid:
                print(" ".join(row))
            print()

    def _level_width(self, level: int) -> int:
        if level == 1:
            return 4
        if level == 2:
            return 8
        if level == 3:
            return 12
        raise ValueError(f"Invalid level: {level}")


# =========================
# Optional smoke test
# =========================
if __name__ == "__main__":
    env = SuperTicTacToeEnv(seed=42, illegal_action_mode="raise")

    obs = env.reset()
    print("Initial observation shape:", obs.shape)
    print("Initial legal actions:", len(env.legal_actions()))
    env.render()

    # sample a few random legal moves
    for t in range(5):
        if env.done:
            break

        legal = env.legal_actions()
        a = int(env.rng.choice(legal))
        result = env.step(a)

        print(f"\nStep {t}")
        print("Chosen action:", a)
        print("Reward:", result.reward)
        print("Done:", result.done)
        print("Info:", result.info)
        env.render()