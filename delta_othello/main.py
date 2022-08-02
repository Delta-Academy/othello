from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

from check_submission import check_submission
from game_mechanics import (
    OthelloEnv,
    choose_move_randomly,
    load_network,
    play_othello_game,
    save_network,
)

TEAM_NAME = "Team Name"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"


def train() -> nn.Module:
    """
    TODO: Write this function to train your algorithm.
    Returns:
        A pytorch network to be used by choose_move. You can architect
        this however you like but your choose_move function must be able
        to use it.
    """
    raise NotImplementedError("You need to implement this function!")


def choose_move(
    state: np.ndarray, neural_network: nn.Module, verbose: bool = False
) -> Optional[Tuple[int, int]]:
    """Called during competitive play.

    It acts greedily given current state of the board and value
    function dictionary. It returns a single move to play.
    Args:
        state: State of the board as a np array. Your pieces are
                1's, the opponent's are -1's and empty are 0's.
        network: The pytorch network output by train().
        verbose: Whether to print debugging information to console.
    Returns:
        position (Tuple | None): The position (row, col) want to place
                        your counter. row and col should be an integer 0 -> 7)
                        where (0,0) is the top left corner and
                        (7,7) is the bottom right corner. You should return
                        None if not legal move is available
    n.b. The None return is not ideal maybe think about how to improve this.
    """
    raise NotImplementedError("You need to implement this function!")


if __name__ == "__main__":

    ## Example workflow, feel free to edit this! ###
    my_network = train()
    save_network(my_network, TEAM_NAME)

    check_submission(
        TEAM_NAME
    )  # <---- Make sure I pass! Or your solution will not work in the tournament!!

    my_network = load_network(TEAM_NAME)

    # Code below plays a single game of othello against a random
    #  opponent, think about how you might want to adapt this to
    #  test the performance of your algorithm.
    def choose_move_no_value_fn(state: np.ndarray) -> Optional[Tuple[int, int]]:
        """The arguments in play_connect_4_game() require functions that only take the state as
        input.

        This converts choose_move() to that format.
        """
        return choose_move(state, my_network)

    play_othello_game(
        your_choose_move=choose_move_no_value_fn,
        opponent_choose_move=choose_move_randomly,
        game_speed_multiplier=1,
        verbose=True,
    )
