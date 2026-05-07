from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gymnasium.wrappers import RecordEpisodeStatistics

from torchrl_env import SingleAgentVsRandomEnv


@dataclass
class TrainConfig:
    seed: int = 123
    device: str = "cpu"

    total_episodes: int = 2000
    max_steps_per_episode: int = 300

    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    replay_size: int = 50000
    min_replay_size: int = 1000
    target_update_freq: int = 500

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20000

    eval_every: int = 100
    eval_episodes: int = 50

    save_dir: str = "checkpoints"


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


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, obs, action, reward, next_obs, done, action_mask, next_action_mask):
        data = (obs, action, reward, next_obs, done, action_mask, next_action_mask)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]
        obs, actions, rewards, next_obs, dones, masks, next_masks = zip(*batch)
        return (
            np.stack(obs),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_obs),
            np.array(dones, dtype=np.float32),
            np.stack(masks),
            np.stack(next_masks),
        )

    def __len__(self):
        return len(self.buffer)


def epsilon_by_step(step: int, cfg: TrainConfig) -> float:
    if step >= cfg.epsilon_decay_steps:
        return cfg.epsilon_end
    ratio = step / cfg.epsilon_decay_steps
    return cfg.epsilon_start + ratio * (cfg.epsilon_end - cfg.epsilon_start)


def select_action(q_net, obs, action_mask, epsilon, device):
    """
    Epsilon-greedy with action masking.
    """
    legal_actions = np.flatnonzero(action_mask > 0)
    if len(legal_actions) == 0:
        raise RuntimeError("No legal actions available.")

    if np.random.rand() < epsilon:
        return int(np.random.choice(legal_actions))

    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    q_values = q_net(obs_t).detach().cpu().numpy()[0]

    # mask illegal actions
    masked_q = np.full_like(q_values, fill_value=-1e9, dtype=np.float32)
    masked_q[legal_actions] = q_values[legal_actions]

    return int(np.argmax(masked_q))


@torch.no_grad()
def evaluate_policy(q_net, cfg: TrainConfig, n_episodes: int = 50):
    env = RecordEpisodeStatistics(
        SingleAgentVsRandomEnv(seed=cfg.seed + 999, opponent_seed=cfg.seed + 1999)
    )
    q_net.eval()

    wins = 0
    losses = 0
    draws = 0
    returns = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=cfg.seed + 10000 + ep)
        done = False
        total_reward = 0.0

        while not done:
            action_mask = info["action_mask"]
            action = select_action(
                q_net=q_net,
                obs=obs,
                action_mask=action_mask,
                epsilon=0.0,
                device=cfg.device,
            )
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        returns.append(total_reward)

        winner = info.get("winner", 0)
        if winner == 1:
            wins += 1
        elif winner == -1:
            losses += 1
        else:
            draws += 1

    q_net.train()
    return {
        "win_rate": wins / n_episodes,
        "loss_rate": losses / n_episodes,
        "draw_rate": draws / n_episodes,
        "avg_return": float(np.mean(returns)),
    }


def train(cfg: TrainConfig):
    os.makedirs(cfg.save_dir, exist_ok=True)

    env = RecordEpisodeStatistics(
        SingleAgentVsRandomEnv(seed=cfg.seed, opponent_seed=cfg.seed + 1)
    )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    q_net = QNetwork().to(cfg.device)
    target_net = QNetwork().to(cfg.device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=cfg.lr)

    from torchrl_replay_buffer import TorchRLReplayBuffer

    replay = TorchRLReplayBuffer(
        capacity=cfg.replay_size,
        device=cfg.device,
    )

    global_step = 0

    for episode in range(1, cfg.total_episodes + 1):
        obs, info = env.reset(seed=cfg.seed + episode)
        action_mask = info["action_mask"]
        done = False
        ep_return = 0.0
        ep_steps = 0

        while not done and ep_steps < cfg.max_steps_per_episode:
            epsilon = epsilon_by_step(global_step, cfg)

            action = select_action(
                q_net=q_net,
                obs=obs,
                action_mask=action_mask,
                epsilon=epsilon,
                device=cfg.device,
            )

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            next_action_mask = next_info["action_mask"]

            replay.push(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                action_mask=action_mask,
                next_action_mask=next_action_mask,
            )

            obs = next_obs
            action_mask = next_action_mask
            ep_return += reward
            ep_steps += 1
            global_step += 1

            # optimization
            if len(replay) >= cfg.min_replay_size:
                (
                    obs_t,
                    act_t,
                    rew_t,
                    next_obs_t,
                    done_t,
                    mask_t,
                    next_mask_t,
                ) = replay.sample(cfg.batch_size)

                act_t = act_t.long().unsqueeze(-1)
                rew_t = rew_t.float()
                done_t = done_t.float()
                next_mask_t = next_mask_t.bool()

                q_values = q_net(obs_t).gather(1, act_t).squeeze(-1)

                # Double DQN style next-action selection
                next_online_q = q_net(next_obs_t)
                next_online_q[~next_mask_t] = -1e9
                next_actions = torch.argmax(next_online_q, dim=1, keepdim=True)

                next_target_q = target_net(next_obs_t).gather(1, next_actions).squeeze(-1)
                target = rew_t + cfg.gamma * (1.0 - done_t) * next_target_q

                loss = nn.functional.mse_loss(q_values, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if global_step % cfg.target_update_freq == 0:
                    target_net.load_state_dict(q_net.state_dict())

        if episode % 20 == 0:
            print(
                f"Episode {episode:4d} | "
                f"return={ep_return:6.2f} | "
                f"steps={ep_steps:3d} | "
                f"buffer={len(replay):5d}"
            )

        if episode % cfg.eval_every == 0:
            stats = evaluate_policy(q_net, cfg, n_episodes=cfg.eval_episodes)
            print(
                f"[Eval @ ep {episode}] "
                f"win={stats['win_rate']:.3f}, "
                f"loss={stats['loss_rate']:.3f}, "
                f"draw={stats['draw_rate']:.3f}, "
                f"avg_return={stats['avg_return']:.3f}"
            )

            ckpt_path = os.path.join(cfg.save_dir, f"dqn_ep{episode}.pt")
            torch.save(
                {
                    "episode": episode,
                    "model_state_dict": q_net.state_dict(),
                    "target_state_dict": target_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg.__dict__,
                },
                ckpt_path,
            )

    final_path = os.path.join(cfg.save_dir, "dqn_final.pt")
    torch.save(
        {
            "episode": cfg.total_episodes,
            "model_state_dict": q_net.state_dict(),
            "target_state_dict": target_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg.__dict__,
        },
        final_path,
    )
    print(f"Training finished. Final model saved to: {final_path}")


if __name__ == "__main__":
    cfg = TrainConfig(
        seed=123,
        device="cuda" if torch.cuda.is_available() else "cpu",
        total_episodes=1000,
        eval_every=100,
        eval_episodes=50,
        min_replay_size=1000,
        batch_size=64,
        replay_size=50000,
        target_update_freq=500,
    )
    train(cfg)