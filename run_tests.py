import numpy as np
from collections import Counter

from env import SuperTicTacToeEnv


def assert_true(cond, msg):
    if not cond:
        raise AssertionError(msg)


# =========================================================
# Basic reset / initialization tests
# =========================================================
def test_reset_initial_state():
    env = SuperTicTacToeEnv(seed=123)
    obs = env.reset()

    assert_true(obs.shape == (97,), "obs shape should be (97,)")
    assert_true(np.all(env.board == 0), "board should be all zeros after reset")
    assert_true(env.current_player == 1, "current_player should be 1 after reset")
    assert_true(env.done is False, "done should be False after reset")
    assert_true(env.winner == 0, "winner should be 0 after reset")
    assert_true(env.move_count == 0, "move_count should be 0 after reset")
    assert_true(env.last_action is None, "last_action should be None after reset")
    assert_true(env.last_realized_action is None, "last_realized_action should be None after reset")
    assert_true(env.last_move_forfeited is False, "last_move_forfeited should be False after reset")
    print("test_reset_initial_state passed!")


def test_initial_action_mask():
    env = SuperTicTacToeEnv(seed=123)
    env.reset()

    mask = env.get_action_mask()
    assert_true(mask.shape == (96,), "action mask shape should be (96,)")
    assert_true(mask.sum() == 96, "all 96 actions should be legal initially")
    assert_true(np.all(mask), "all entries in action mask should be True initially")
    print("test_initial_action_mask passed!")


def test_legal_actions_after_reset():
    env = SuperTicTacToeEnv(seed=123)
    env.reset()

    legal = env.legal_actions()
    assert_true(len(legal) == 96, "there should be 96 legal actions after reset")
    assert_true(legal == list(range(96)), "initial legal actions should be 0..95")
    print("test_legal_actions_after_reset passed!")


# =========================================================
# Action legality tests
# =========================================================
def test_is_legal_action_rejects_invalid_inputs():
    env = SuperTicTacToeEnv(seed=123)
    env.reset()

    bad_actions = [-1, 96, 999, "abc", None]
    for bad_action in bad_actions:
        assert_true(env.is_legal_action(bad_action) is False, f"{bad_action} should be illegal")
    print("test_is_legal_action_rejects_invalid_inputs passed!")


def test_is_legal_action_rejects_occupied_square():
    env = SuperTicTacToeEnv(seed=123)
    env.reset()

    env.board[10] = 1
    assert_true(env.is_legal_action(10) is False, "occupied square should be illegal")
    print("test_is_legal_action_rejects_occupied_square passed!")


# =========================================================
# Step tests
# =========================================================
def test_step_successful_realized_move():
    env = SuperTicTacToeEnv(seed=123)
    env.reset()

    env.resolve_action = lambda action: action

    result = env.step(10)

    assert_true(result.done is False, "game should not end after one normal move")
    assert_true(result.reward == 0.0, "nonterminal reward should be 0")
    assert_true(env.board[10] == 1, "board[10] should be occupied by player 1")
    assert_true(env.move_count == 1, "move_count should become 1")
    assert_true(env.current_player == -1, "player should switch after successful move")
    assert_true(result.info["realized_action"] == 10, "realized action should be 10")
    assert_true(result.info["move_forfeited"] is False, "move should not be forfeited")
    print("test_step_successful_realized_move passed!")


def test_step_forfeited_realized_move():
    env = SuperTicTacToeEnv(seed=123)
    env.reset()

    env.resolve_action = lambda action: None
    current_player_before = env.current_player

    result = env.step(10)

    assert_true(result.done is False, "game should not end after forfeited move")
    assert_true(result.reward == 0.0, "forfeited nonterminal reward should be 0")
    assert_true(np.count_nonzero(env.board) == 0, "board should remain unchanged after forfeited move")
    assert_true(env.move_count == 0, "move_count should remain 0 after forfeited move")
    assert_true(env.current_player == -current_player_before, "player should still switch after forfeited move")
    assert_true(result.info["move_forfeited"] is True, "move should be marked forfeited")
    assert_true(result.info["realized_action"] is None, "realized action should be None")
    print("test_step_forfeited_realized_move passed!")


def test_illegal_action_forfeit_mode():
    env = SuperTicTacToeEnv(seed=123, illegal_action_mode="forfeit")
    env.reset()

    env.board[5] = 1
    current_player_before = env.current_player

    result = env.step(5)

    assert_true(result.done is False, "game should not end")
    assert_true(result.reward == 0.0, "reward should be 0")
    assert_true(result.info["move_forfeited"] is True, "illegal action should be forfeited in forfeit mode")
    assert_true(result.info["realized_action"] is None, "realized action should be None")
    assert_true(env.current_player == -current_player_before, "player should switch")
    print("test_illegal_action_forfeit_mode passed!")


# =========================================================
# resolve_action probability tests
# =========================================================
def test_resolve_action_center_distribution():
    env = SuperTicTacToeEnv(seed=123)
    env.reset()

    action = 69  # interior point in level 3
    trials = 20000
    counts = Counter()

    for _ in range(trials):
        realized = env.resolve_action(action)
        counts[realized] += 1

    intended_prob = counts[action] / trials
    assert_true(abs(intended_prob - 0.5) < 0.03, f"intended probability off: {intended_prob}")

    neighbors = env.geometry.neighbors[action]
    assert_true(all(n is not None for n in neighbors), "all 8 neighbors should be valid for interior point")

    for n in neighbors:
        p = counts[n] / trials
        assert_true(abs(p - 1.0 / 16.0) < 0.02, f"neighbor probability off for {n}: {p}")

    assert_true(counts[None] == 0, "interior point should not forfeit due to outside board")
    print("test_resolve_action_center_distribution passed!")


def test_resolve_action_corner_outside_probability():
    env = SuperTicTacToeEnv(seed=123)
    env.reset()

    action = 0  # top-left corner of level 1
    trials = 20000
    counts = Counter()

    for _ in range(trials):
        realized = env.resolve_action(action)
        counts[realized] += 1

    outside_prob = counts[None] / trials
    assert_true(abs(outside_prob - 5.0 / 16.0) < 0.03, f"outside probability off: {outside_prob}")
    print("test_resolve_action_corner_outside_probability passed!")


def test_resolve_action_forfeit_when_sampled_square_is_occupied():
    env = SuperTicTacToeEnv(seed=123)
    env.reset()

    action = 69

    # 先占用 intended action，自然只要采样到它就必须 forfeited
    env.board[action] = -1

    realized = env.resolve_action(action)

    # 这里不一定每次都采到 intended，因此不能一次就断言
    # 我们多跑几次，只要出现 None 就说明 occupied sampled square 会 forfeited
    found_none = realized is None

    for _ in range(200):
        realized = env.resolve_action(action)
        if realized is None:
            found_none = True
            break

    assert_true(found_none, "occupied sampled square should eventually cause forfeited move")
    print("test_resolve_action_forfeit_when_sampled_square_is_occupied passed!")


# =========================================================
# Winner tests
# =========================================================
def test_check_winner_row_player1():
    env = SuperTicTacToeEnv(seed=123)
    env.reset()

    env.board[[0, 1, 2, 3]] = 1
    assert_true(env.check_winner() == 1, "player 1 should win on row [0,1,2,3]")
    print("test_check_winner_row_player1 passed!")


def test_check_winner_row_player_minus1():
    env = SuperTicTacToeEnv(seed=123)
    env.reset()

    env.board[[16, 17, 18, 19]] = -1
    assert_true(env.check_winner() == -1, "player -1 should win on row [16,17,18,19]")
    print("test_check_winner_row_player_minus1 passed!")


def test_check_winner_no_false_positive():
    env = SuperTicTacToeEnv(seed=123)
    env.reset()

    env.board[[0, 1, 2]] = 1
    assert_true(env.check_winner() == 0, "3 in a row should not be a win")

    env.board[3] = -1
    assert_true(env.check_winner() == 0, "mixed row should not be a win")
    print("test_check_winner_no_false_positive passed!")


def test_step_terminal_win():
    env = SuperTicTacToeEnv(seed=123)
    env.reset()

    env.board[[0, 1, 2]] = 1
    env.resolve_action = lambda action: action

    result = env.step(3)

    assert_true(result.done is True, "game should end with winning move")
    assert_true(result.reward == 1.0, "winning reward should be 1")
    assert_true(env.done is True, "env.done should be True")
    assert_true(env.winner == 1, "winner should be player 1")
    assert_true(result.info["winner"] == 1, "info winner should be 1")
    print("test_step_terminal_win passed!")

def test_check_winner_diagonal_player1():
    env = SuperTicTacToeEnv(seed=123)
    env.reset()

    assert_true(len(env.geometry.diag_lines) > 0, "diag_lines should not be empty")

    line = env.geometry.diag_lines[0]
    assert_true(len(line) == 5, "a diagonal winning line should have length 5")

    env.board[line] = 1
    assert_true(env.check_winner() == 1, f"player 1 should win on diagonal line {line}")
    print("test_check_winner_diagonal_player1 passed!")


def test_check_winner_column_player_minus1():
    env = SuperTicTacToeEnv(seed=123)
    env.reset()

    assert_true(len(env.geometry.column_lines) > 0, "column_lines should not be empty")

    line = env.geometry.column_lines[0]
    assert_true(len(line) == 4, "a column winning line should have length 4")

    env.board[line] = -1
    assert_true(env.check_winner() == -1, f"player -1 should win on column line {line}")
    print("test_check_winner_column_player_minus1 passed!")


def test_check_winner_diagonal_no_false_positive():
    env = SuperTicTacToeEnv(seed=123)
    env.reset()

    assert_true(len(env.geometry.diag_lines) > 0, "diag_lines should not be empty")

    line = env.geometry.diag_lines[0]
    assert_true(len(line) == 5, "a diagonal winning line should have length 5")

    env.board[line[:4]] = 1
    assert_true(env.check_winner() == 0, "4 out of 5 on a diagonal should not be a win")
    print("test_check_winner_diagonal_no_false_positive passed!")


def test_check_winner_column_no_false_positive():
    env = SuperTicTacToeEnv(seed=123)
    env.reset()

    assert_true(len(env.geometry.column_lines) > 0, "column_lines should not be empty")

    line = env.geometry.column_lines[0]
    assert_true(len(line) == 4, "a column winning line should have length 4")

    env.board[line[:3]] = -1
    assert_true(env.check_winner() == 0, "3 out of 4 on a column should not be a win")
    print("test_check_winner_column_no_false_positive passed!")


# =========================================================
# Draw tests
# =========================================================
def test_is_draw_false_when_empty():
    env = SuperTicTacToeEnv(seed=123)
    env.reset()

    assert_true(env.is_draw() is False, "empty board should not be draw")
    print("test_is_draw_false_when_empty passed!")


def test_is_draw_false_when_space_remains():
    env = SuperTicTacToeEnv(seed=123)
    env.reset()

    env.board[:] = 1
    env.board[10] = 0
    assert_true(env.is_draw() is False, "board with empty square should not be draw")
    print("test_is_draw_false_when_space_remains passed!")


def test_is_draw_false_when_winner_exists():
    env = SuperTicTacToeEnv(seed=123)
    env.reset()

    env.board[:] = 1
    assert_true(env.is_draw() is False, "full winning board should not be draw")
    print("test_is_draw_false_when_winner_exists passed!")


# =========================================================
# Action mask / observation tests
# =========================================================
def test_action_mask_updates_after_successful_step():
    env = SuperTicTacToeEnv(seed=123)
    env.reset()

    env.resolve_action = lambda action: action
    env.step(10)

    mask = env.get_action_mask()
    assert_true(not bool(mask[10]), "mask[10] should become False after occupying square 10")
    assert_true(mask.sum() == 95, "legal action count should become 95")
    print("test_action_mask_updates_after_successful_step passed!")


def test_action_mask_unchanged_after_forfeit():
    env = SuperTicTacToeEnv(seed=123)
    env.reset()

    before = env.get_action_mask().copy()
    env.resolve_action = lambda action: None
    env.step(10)
    after = env.get_action_mask()

    assert_true(np.array_equal(before, after), "mask should remain unchanged after forfeited move")
    print("test_action_mask_unchanged_after_forfeit passed!")


def test_observation_last_entry_is_current_player():
    env = SuperTicTacToeEnv(seed=123)
    obs = env.reset()
    assert_true(obs[96] == 1, "initial current_player in observation should be 1")

    env.resolve_action = lambda action: action
    result = env.step(10)
    assert_true(result.observation[96] == -1, "current_player in next observation should be -1")
    print("test_observation_last_entry_is_current_player passed!")

from agents.heuristic_agent import HeuristicAgent
from env import SuperTicTacToeEnv


def test_heuristic_agent_uses_current_player_perspective():
    env = SuperTicTacToeEnv(seed=123)
    env.reset()

    agent = HeuristicAgent(seed=123)

    # Make it player -1's turn
    env.current_player = -1

    # Give player -1 three in a row and one winning square
    env.board[[0, 1, 2]] = -1

    action = agent.act(env)

    assert action == 3, f"Expected player -1 to choose winning action 3, got {action}"
    print("test_heuristic_agent_uses_current_player_perspective passed!")


# =========================================================
# Run all tests
# =========================================================
def run_all_tests():
    test_reset_initial_state()
    test_initial_action_mask()
    test_legal_actions_after_reset()
    test_is_legal_action_rejects_invalid_inputs()
    test_is_legal_action_rejects_occupied_square()

    test_step_successful_realized_move()
    test_step_forfeited_realized_move()
    test_illegal_action_forfeit_mode()

    test_resolve_action_center_distribution()
    test_resolve_action_corner_outside_probability()
    test_resolve_action_forfeit_when_sampled_square_is_occupied()

    test_check_winner_row_player1()
    test_check_winner_row_player_minus1()
    test_check_winner_no_false_positive()

    test_check_winner_diagonal_player1()
    test_check_winner_column_player_minus1()
    test_check_winner_diagonal_no_false_positive()
    test_check_winner_column_no_false_positive()

    test_step_terminal_win()

    test_is_draw_false_when_empty()
    test_is_draw_false_when_space_remains()
    test_is_draw_false_when_winner_exists()

    test_action_mask_updates_after_successful_step()
    test_action_mask_unchanged_after_forfeit()
    test_observation_last_entry_is_current_player()
    test_heuristic_agent_uses_current_player_perspective()

    print("\nAll tests passed!")


if __name__ == "__main__":
    run_all_tests()