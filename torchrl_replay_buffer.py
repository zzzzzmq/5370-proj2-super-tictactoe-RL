from __future__ import annotations

import torch
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage


class TorchRLReplayBuffer:
    """
    TorchRL-based replay buffer wrapper.

    This class uses TensorDictReplayBuffer and LazyTensorStorage from TorchRL.
    It keeps the same push/sample style as the original custom replay buffer,
    so the training script only needs small modifications.
    """

    def __init__(self, capacity: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = torch.device(device)

        self.buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=capacity),
            batch_size=None,
        )

    def push(
        self,
        obs,
        action,
        reward,
        next_obs,
        done,
        action_mask,
        next_action_mask,
    ) -> None:
        td = TensorDict(
            {
                "obs": torch.tensor(obs, dtype=torch.float32),
                "action": torch.tensor(action, dtype=torch.long),
                "reward": torch.tensor(reward, dtype=torch.float32),
                "next_obs": torch.tensor(next_obs, dtype=torch.float32),
                "done": torch.tensor(done, dtype=torch.bool),
                "action_mask": torch.tensor(action_mask, dtype=torch.bool),
                "next_action_mask": torch.tensor(next_action_mask, dtype=torch.bool),
            },
            batch_size=[],
        )

        self.buffer.add(td)

    def sample(self, batch_size: int):
        td = self.buffer.sample(batch_size).to(self.device)

        obs = td["obs"]
        actions = td["action"]
        rewards = td["reward"]
        next_obs = td["next_obs"]
        dones = td["done"].float()
        masks = td["action_mask"]
        next_masks = td["next_action_mask"]

        return obs, actions, rewards, next_obs, dones, masks, next_masks

    def __len__(self) -> int:
        return len(self.buffer)