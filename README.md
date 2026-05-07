# Super Tic-Tac-Toe Reinforcement Learning Environment

This project implements a stochastic super tic-tac-toe game as a reinforcement learning environment. It includes game rule formalization, environment construction, baseline agents, TorchRL-based DQN training, result visualization, and a lightweight interactive demo.

## Project Structure

- `geometry.py`: Board geometry, coordinate mappings, neighbors, and winning-line enumeration.
- `env.py`: Main stochastic game environment.
- `agents/`: Random and heuristic baseline agents.
- `evaluate_baselines.py`: Baseline matchup evaluation.
- `torchrl_env.py`: Single-agent wrapper against a random opponent.
- `torchrl_replay_buffer.py`: TorchRL replay buffer using `TensorDictReplayBuffer`.
- `train_torchrl_dqn.py`: DQN training script.
- `plot_results.py`: Generate result figures.
- `run_tests.py`: Environment and geometry tests.
- `watch_game.py`: Terminal-based game watching script.
- `interactive_app.py`: Lightweight Streamlit interactive demo.
- `figures/`: Result figures and demo screenshots.
- `5370_proj2_report.pdf`: Final project report.

## Installation

```bash
pip install numpy matplotlib torch torchrl tensordict streamlit
```

## Run Tests

```bash
python run_tests.py
```

## Evaluate Baselines

```bash
python evaluate_baselines.py
```

## Train DQN Agent

```bash
python train_torchrl_dqn.py
```

## Generate Figures

```bash
python plot_results.py
```

## Watch One Game in Terminal

```bash
python watch_game.py
```

## Interactive Demo

```bash
streamlit run interactive_app.py
```

## Main Results

Baseline matchup results over 200 games:

| Matchup | P1 Win | P2 Win | Draw | Avg Move Count | Avg Turns |
|---|---:|---:|---:|---:|---:|
| Random vs Random | 0.545 | 0.455 | 0.000 | 50.77 | 68.00 |
| Heuristic vs Random | 0.960 | 0.040 | 0.000 | 31.21 | 39.55 |
| Heuristic vs Heuristic | 0.565 | 0.435 | 0.000 | 31.91 | 40.49 |

The DQN agent reaches a best evaluation win rate of 0.70 against the random opponent during training, while the final evaluation win rate is 0.54.

## Report

The final report is included as:

```text
5370_proj2_report.pdf
```
