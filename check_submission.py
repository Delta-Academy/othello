from pathlib import Path
from typing import Optional, Tuple
from torch import nn

import delta_utils.check_submission as checker
from game_mechanics import get_empty_board, load_network


def check_submission(team_name: str) -> None:
    example_state = get_empty_board()
    expected_choose_move_return_type = Optional[Tuple]
    game_mechanics_expected_hash = (
        "bee9b2a3bec41fb533a51d3b4acca623e2d2cc88cfd309a2edf214330192ea5f"
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
