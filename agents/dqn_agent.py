# agents/dqn_agent.py
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int = 97, action_dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        model_path: str = "checkpoints/dqn_final.pt",
        device: Optional[str] = None,
        name: str = "DQNAgent",
    ) -> None:
        self.name = name

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.q_net = QNetwork().to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"DQN checkpoint not found: {model_path}\n"
                f"Please train the model first or provide a valid path."
            )

        checkpoint = torch.load(model_path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["model_state_dict"])
        self.q_net.eval()

    def act(self, env) -> int:
        obs = env.get_observation()
        action_mask = env.get_action_mask()
        legal_actions = np.flatnonzero(action_mask > 0)

        if len(legal_actions) == 0:
            raise RuntimeError("No legal actions available.")

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_net(obs_t).cpu().numpy()[0]

        masked_q = np.full_like(q_values, fill_value=-1e9, dtype=np.float32)
        masked_q[legal_actions] = q_values[legal_actions]

        return int(np.argmax(masked_q))