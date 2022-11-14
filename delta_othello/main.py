from typing import Optional, Tuple

import numpy as np
from torch import nn

from check_submission import check_submission
from game_mechanics import (
    OthelloEnv,
    choose_move_randomly,
    get_legal_moves,
    human_player,
    is_terminal,
    load_network,
    make_move,
    play_othello_game,
    reward_function,
    save_network,
)

TEAM_NAME = "Team Name"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"


def train() -> nn.Module:
    """
    TODO: Write this function to train your network.
    """
    raise NotImplementedError("You need to implement this function!")


def choose_move(
    state: np.ndarray,
    neural_network: nn.Module,
) -> Optional[Tuple[int, int]]:
    """Called during competitive play.

    It acts greedily given current state of the board and
    your neural_netwok. It returns a single move to play.
    Args:
        state: [6x6] np array defining state of the board
        neural_network: Your network from train()
    Returns:
        move: (row, col) position to place your piece OR None if no legal moves
    """
    raise NotImplementedError("You need to implement this function!")


if __name__ == "__main__":

    # ## Example workflow, feel free to edit this! ###
    neural_network = train()
    save_network(neural_network, TEAM_NAME)

    check_submission(
        TEAM_NAME
    )  # <---- Make sure I pass! Or your solution will not work in the tournament!!

    my_network = load_network(TEAM_NAME)

    # # Code below plays a single game against a random
    # #  opponent, think about how you might want to adapt this to
    # #  test the performance of your algorithm.
    def choose_move_no_network(state: np.ndarray) -> Optional[Tuple[int, int]]:
        """The arguments in play_game() require functions that only take the state as input.

        This converts choose_move() to that format.
        """
        return choose_move(state, my_network)

    # Play a game against your bot! You play with the black counters.
    # Click a grey circle to place a piece.
    play_othello_game(
        your_choose_move=human_player,
        opponent_choose_move=choose_move_no_network,
        game_speed_multiplier=10,
        verbose=True,
        render=True,
    )
