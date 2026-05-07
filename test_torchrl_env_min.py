from torchrl_env import SingleAgentVsRandomEnv

env = SingleAgentVsRandomEnv(seed=123, opponent_seed=456)

obs, info = env.reset()
print("reset ok")
print("obs shape:", obs.shape)
print("action_mask shape:", info["action_mask"].shape)
print("current_player:", info["current_player"])
print("winner:", info["winner"])
print("done:", info["done"])
print("move_count:", info["move_count"])

legal_actions = [i for i, v in enumerate(info["action_mask"]) if v == 1]
print("num legal actions at reset:", len(legal_actions))

action = legal_actions[0]
print("taking action:", action)

next_obs, reward, terminated, truncated, next_info = env.step(action)

print("\nstep ok")
print("next_obs shape:", next_obs.shape)
print("reward:", reward)
print("terminated:", terminated)
print("truncated:", truncated)
print("stage:", next_info["stage"])
print("current_player after step:", next_info["current_player"])
print("winner after step:", next_info["winner"])
print("move_count after step:", next_info["move_count"])
print("agent realized action:", next_info.get("agent_realized_action"))
print("agent move forfeited:", next_info.get("agent_move_forfeited"))
print("opponent action:", next_info.get("opponent_action"))
print("opponent realized action:", next_info.get("opponent_realized_action"))
print("opponent move forfeited:", next_info.get("opponent_move_forfeited"))

print("\nrender current board:")
env.render()