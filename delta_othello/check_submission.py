import itertools
from pathlib import Path
from typing import List, Optional, Tuple

from delta_utils import get_discrete_choose_move_out_checker
from delta_utils.check_submission import check_submission as _check_submission
from game_mechanics import get_empty_board, load_network


def check_submission(team_name: str) -> None:
    example_state = get_empty_board()
    user_pkl_file = load_network(team_name)

    board_dim = example_state.shape[0]
    possible_outputs: List[Optional[Tuple[int, int]]] = list(
        itertools.product(range(board_dim), range(board_dim))
    )
    possible_outputs.append(None)

    _check_submission(
        example_choose_move_input={"state": example_state, "neural_network": user_pkl_file},
        check_choose_move_output=get_discrete_choose_move_out_checker(
            possible_outputs=possible_outputs,
        ),
        current_folder=Path(__file__).parent.resolve(),
    )
