from __future__ import annotations

import streamlit as st
import numpy as np
from env import SuperTicTacToeEnv

try:
    from agents.random_agent import RandomAgent
    from agents.heuristic_agent import HeuristicAgent
    from agents.dqn_agent import DQNAgent          
except ModuleNotFoundError:
    from random_agent import RandomAgent
    from heuristic_agent import HeuristicAgent
    from dqn_agent import DQNAgent                 


def idx_to_lbrc(index: int):
    """
    Convert flat index to (level, block, row, col).

    Level 1: indices 0-15
    Level 2: indices 16-47
    Level 3: indices 48-95
    """
    if 0 <= index < 16:
        level = 1
        local = index
        block = local // 16
    elif 16 <= index < 48:
        level = 2
        local = index - 16
        block = local // 16
    elif 48 <= index < 96:
        level = 3
        local = index - 48
        block = local // 16
    else:
        raise ValueError(f"Invalid index: {index}")

    inside = local % 16
    row = inside // 4
    col = inside % 4
    return level, block, row, col


def lbrc_to_idx(level: int, block: int, row: int, col: int) -> int:
    if level == 1:
        return 16 * block + 4 * row + col
    if level == 2:
        return 16 + 16 * block + 4 * row + col
    if level == 3:
        return 48 + 16 * block + 4 * row + col
    raise ValueError(f"Invalid level: {level}")


def get_level_width(level: int) -> int:
    if level == 1:
        return 4
    if level == 2:
        return 8
    if level == 3:
        return 12
    raise ValueError(f"Invalid level: {level}")


def display_symbol(value: int) -> str:
    if value == 1:
        return "O"
    if value == -1:
        return "X"
    return "·"


def parse_step_result(result):
    """
    Your env.step returns StepResult with:
    observation, reward, done, info.
    """
    return result.observation, result.reward, result.done, result.info


def make_opponent(name: str, seed: int, model_path: str = "checkpoints/dqn_final.pt"):
    if name == "RandomAgent":
        return RandomAgent(seed=seed)
    if name == "HeuristicAgent":
        return HeuristicAgent(seed=seed)
    if name == "DQNAgent":
        return DQNAgent(model_path=model_path)    
    raise ValueError(f"Unknown opponent: {name}")


def reset_game(opponent_name: str, seed: int, model_path: str = "checkpoints/dqn_final.pt"):
    env = SuperTicTacToeEnv(seed=seed)
    env.reset()

    st.session_state.env = env
    st.session_state.opponent_name = opponent_name
    st.session_state.opponent = make_opponent(opponent_name, seed + 100, model_path)
    st.session_state.logs = []
    st.session_state.game_seed = seed


def append_log(text: str):
    st.session_state.logs.append(text)


def step_human_action(action: int):
    env = st.session_state.env
    opponent = st.session_state.opponent

    if env.done:
        return

    if env.current_player != 1:
        append_log("It is not Human Player's turn.")
        return

    result = env.step(action)
    _, reward, done, info = parse_step_result(result)

    append_log(
        f"Human intended {info.get('intended_action')}, "
        f"realized {info.get('realized_action')}, "
        f"forfeited={info.get('move_forfeited')}, "
        f"reward={reward}"
    )

    if env.done:
        return

    # Opponent automatically moves.
    if env.current_player == -1:
        opp_action = opponent.act(env)
        result = env.step(opp_action)
        _, reward, done, info = parse_step_result(result)

        append_log(
            f"{st.session_state.opponent_name} intended {info.get('intended_action')}, "
            f"realized {info.get('realized_action')}, "
            f"forfeited={info.get('move_forfeited')}, "
            f"reward={reward}"
        )

def render_cell(container, idx: int, value: int) -> None:
    """
    Render one board cell.

    Empty cells are clickable buttons.
    Occupied cells are rendered as colored HTML cards,
    because disabled Streamlit buttons look too faint.
    """
    env = st.session_state.env

    if value == 0:
        disabled = env.done or env.current_player != 1

        if container.button(
            f"·\n{idx}",
            key=f"cell_{idx}",
            disabled=disabled,
            use_container_width=True,
        ):
            step_human_action(idx)
            st.rerun()

    elif value == 1:
        container.markdown(
            f"""
            <div style="
                width: 100%;
                aspect-ratio: 1 / 1;
                border-radius: 6px;
                border: 2px solid #1f77b4;
                background-color: #d8ecff;
                color: #0b4f8a;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                font-weight: 700;
                font-size: 16px;
                line-height: 1.0;
                margin-bottom: 0.35rem;
            ">
                <div>O</div>
                <div style="font-size: 9px; font-weight: 500; opacity: 0.75;">{idx}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif value == -1:
        container.markdown(
            f"""
            <div style="
                width: 100%;
                aspect-ratio: 1 / 1;
                border-radius: 6px;
                border: 2px solid #d62728;
                background-color: #ffe0e0;
                color: #9c1111;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                font-weight: 700;
                font-size: 16px;
                line-height: 1.0;
                margin-bottom: 0.35rem;
            ">
                <div>X</div>
                <div style="font-size: 9px; font-weight: 500; opacity: 0.75;">{idx}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def render_level(level: int):
    env = st.session_state.env
    width = get_level_width(level)

    MAX_WIDTH = 12  
    left_pad = (MAX_WIDTH - width) // 2  

    # st.markdown(f"### Level {level}")

    for row in range(4):
        cols = st.columns(MAX_WIDTH)
        for gc in range(width):
            block = gc // 4
            col = gc % 4
            idx = lbrc_to_idx(level, block, row, col)
            value = int(env.board[idx])

            # render_cell(cols[gc], idx, value)
            render_cell(cols[gc + left_pad], idx, value)


def render_status():
    env = st.session_state.env

    st.subheader("Game Status")

    if env.winner == 1:
        winner_text = "Human Player wins (O)"
    elif env.winner == -1:
        winner_text = f"{st.session_state.opponent_name} wins (X)"
    elif env.done:
        winner_text = "Draw"
    else:
        winner_text = "No winner yet"

    current = "Human Player (O)" if env.current_player == 1 else f"{st.session_state.opponent_name} (X)"

    st.write(f"**Current player:** {current}")
    st.write(f"**Done:** {env.done}")
    st.write(f"**Winner:** {winner_text}")
    st.write(f"**Successful move count:** {env.move_count}")

    if hasattr(env, "last_action"):
        st.write(f"**Last intended action:** {env.last_action}")
    if hasattr(env, "last_realized_action"):
        st.write(f"**Last realized action:** {env.last_realized_action}")
    if hasattr(env, "last_move_forfeited"):
        st.write(f"**Last move forfeited:** {env.last_move_forfeited}")

    st.markdown("---")
    st.markdown("**Legend**")
    st.markdown(
        """
        <div style="display:flex; gap:10px;">
            <div style="padding:8px 14px; border-radius:8px; background:#d8ecff; color:#0b4f8a; border:2px solid #1f77b4; font-weight:700;">O Human / Player 1</div>
            <div style="padding:8px 14px; border-radius:8px; background:#ffe0e0; color:#9c1111; border:2px solid #d62728; font-weight:700;">X Agent / Player 2</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def main():
    st.set_page_config(
        page_title="Super Tic-Tac-Toe RL Demo",
        layout="wide",
    )

    st.title("Super Tic-Tac-Toe Interactive Demo")
    st.caption("Human Player (O) vs Agent Opponent (X)")

    st.markdown("""
    <style>
        div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button {
            width: 100% !important;
            height: auto !important;
            aspect-ratio: 1 / 1 !important;
            padding: 1px !important;
            min-height: 0px !important;
            font-size: 9px !important;
            line-height: 1.1 !important;
        }
        div[data-testid="stHorizontalBlock"] div[data-testid="column"] {
            padding-left: 2px !important;
            padding-right: 2px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # ========== Sidebar ==========
    with st.sidebar:
        st.header("Settings")

        opponent_name = st.selectbox(
            "Opponent",
            ["RandomAgent", "HeuristicAgent", "DQNAgent"],
            index=0,
        )

        model_path = "checkpoints/dqn_final.pt"
        if opponent_name == "DQNAgent":
            model_path = st.text_input(
                "Model checkpoint path",
                value="checkpoints/dqn_final.pt",
                help="Path to a saved DQN checkpoint (.pt file)",
            )

        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=999999,
            value=123,
            step=1,
        )

        if st.button("New Game", use_container_width=True):
            reset_game(opponent_name, int(seed), model_path)
            st.rerun()

        st.markdown("---")
        st.write("Rule reminder:")
        st.write("- Intended square: probability 1/2")
        st.write("- Each adjacent square: probability 1/16")
        st.write("- Outside or occupied realized square: forfeited move")

    # ========== 初始化游戏状态 ==========
    if "env" not in st.session_state:
        reset_game(opponent_name, int(seed), model_path)

    # ========== 页面主体布局 ==========
    left, right = st.columns([2.2, 1.0])

    with left:
        render_level(1)
        render_level(2)
        render_level(3)

    with right:
        render_status()

        st.subheader("Move Log")
        if len(st.session_state.logs) == 0:
            st.write("No moves yet.")
        else:
            for line in reversed(st.session_state.logs[-15:]):
                st.text(line)


if __name__ == "__main__":
    main()