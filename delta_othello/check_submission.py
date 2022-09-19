from pathlib import Path
from typing import Optional, Tuple

import delta_utils.check_submission as checker
from torch import nn

from game_mechanics import get_empty_board, load_network


def check_submission(team_name: str) -> None:
    example_state = get_empty_board()
    expected_choose_move_return_type = tuple
    game_mechanics_expected_hash = (
        "91f3f4308395f327e3e578d673e931feaaec9bf33ff4ead7b274ce8a2b701640"
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
