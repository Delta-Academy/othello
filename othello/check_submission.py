from pathlib import Path
from typing import Optional, Tuple

import delta_utils.check_submission as checker
from torch import nn

from game_mechanics import get_empty_board, load_network


def check_submission(team_name: str) -> None:
    example_state = get_empty_board()
    expected_choose_move_return_type = tuple
    game_mechanics_expected_hash = (
        "8ae6b7431b6c5767d8c10ac9bfc4c72059529172a92c3ad4571e36b77d0ead2a"
    )
    expected_pkl_output_type = nn.Module

    return checker.check_submission(
        example_state=example_state,
        expected_choose_move_return_type=expected_choose_move_return_type,
        expected_pkl_type=expected_pkl_output_type,
        pkl_file=load_network(team_name),
        pkl_checker_function=lambda x: x,
        game_mechanics_hash=game_mechanics_expected_hash,
        current_folder=Path(__file__).parent.resolve(),
    )
