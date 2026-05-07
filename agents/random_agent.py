from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RandomAgent:
    """
    A simple random baseline agent.

    It samples uniformly from the current legal action set.
    """
    seed: Optional[int] = None
    name: str = "RandomAgent"

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def act(self, env) -> int:
        """
        Choose one legal action uniformly at random.

        Args:
            env: environment instance with legal_actions()

        Returns:
            action (int)
        """
        legal = env.legal_actions()
        if len(legal) == 0:
            raise RuntimeError(f"{self.name} found no legal actions to play.")
        return int(self.rng.choice(legal))