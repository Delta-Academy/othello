import random
from typing import Optional, Tuple

import numpy as np
import pytest

from delta_othello.game_mechanics import OthelloEnv, _get_legal_moves, has_legal_move, is_legal_move


def choose_move4x4(state: np.ndarray) -> Optional[Tuple[int, int]]:
    board = np.array(
        [
            [0] * 4,
            [0, -1, 1, 0],
            [0, 1, -1, 0],
            [0] * 4,
        ]
    )
    if np.array_equal(state, board) or np.array_equal(state, board * -1):
        return (3, 2)

    board = np.array(
        [
            [0] * 4,
            [0, -1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 0],
        ]
    )

    if np.array_equal(state, board) or np.array_equal(state, board * -1):
        return (1, 3)

    board = np.array(
        [
            [0, 0, 0, 0],
            [0, -1, -1, -1],
            [0, 1, 1, 0],
            [0, 0, 1, 0],
        ]
    )

    if np.array_equal(state, board) or np.array_equal(state, board * -1):
        return (0, 0)

    board = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, -1, -1],
            [0, 1, 1, 0],
            [0, 0, 1, 0],
        ]
    )

    if np.array_equal(state, board) or np.array_equal(state, board * -1):
        return (1, 0)

    board = np.array(
        [
            [1, 0, 0, 0],
            [-1, -1, -1, -1],
            [0, 1, 1, 0],
            [0, 0, 1, 0],
        ]
    )

    if np.array_equal(state, board) or np.array_equal(state, board * -1):
        return (0, 1)

    board = np.array(
        [
            [1, 1, 0, 0],
            [-1, 1, -1, -1],
            [0, 1, 1, 0],
            [0, 0, 1, 0],
        ]
    )

    if np.array_equal(state, board) or np.array_equal(state, board * -1):
        return (3, 1)
    board = np.array(
        [
            [1, 1, 0, 0],
            [-1, 1, -1, -1],
            [0, 1, -1, 0],
            [0, -1, 1, 0],
        ]
    )

    if np.array_equal(state, board) or np.array_equal(state, board * -1):
        return (0, 2)

    board = np.array(
        [
            [1, 1, 1, 0],
            [-1, 1, 1, -1],
            [0, 1, 1, 0],
            [0, -1, 1, 0],
        ]
    )

    if np.array_equal(state, board) or np.array_equal(state, board * -1):
        return (3, 3)

    board = np.array(
        [
            [1, 1, 1, 0],
            [-1, 1, 1, -1],
            [0, 1, 1, 0],
            [0, -1, -1, -1],
        ]
    )

    if np.array_equal(state, board) or np.array_equal(state, board * -1):
        return (2, 0)

    return None


def test_4x4() -> None:
    """Full game on a 4x4 board."""
    random.seed(9)  # Means player1 always starts
    game = OthelloEnv(opponent_choose_move=choose_move4x4, board_dim=4)
    game.reset()
    expected = np.array(
        [
            [0] * 4,
            [0, -1, 1, 0],
            [0, 1, -1, 0],
            [0] * 4,
        ]
    )
    np.testing.assert_array_equal(game._board, expected)

    game.step(choose_move4x4(game._board))
    assert not game.game_over
    expected = np.array(
        [
            [0, 0, 0, 0],
            [0, -1, -1, -1],
            [0, 1, 1, 0],
            [0, 0, 1, 0],
        ]
    )
    np.testing.assert_array_equal(game._board, expected)
    game.step(choose_move4x4(game._board))
    assert not game.game_over
    expected = np.array(
        [
            [1, 0, 0, 0],
            [-1, -1, -1, -1],
            [0, 1, 1, 0],
            [0, 0, 1, 0],
        ]
    )
    np.testing.assert_array_equal(game._board, expected)

    assert not game.game_over
    game.step(choose_move4x4(game._board))
    expected = np.array(
        [
            [1, 1, 0, 0],
            [-1, 1, -1, -1],
            [0, 1, -1, 0],
            [0, -1, 1, 0],
        ]
    )
    np.testing.assert_array_equal(game._board, expected)

    assert not game.game_over
    game.step(choose_move4x4(game._board))
    expected = np.array(
        [
            [1, 1, 1, 0],
            [-1, 1, 1, -1],
            [0, 1, 1, 0],
            [0, -1, -1, -1],
        ]
    )
    np.testing.assert_array_equal(game._board, expected)

    game.step(choose_move4x4(game._board))

    assert not _get_legal_moves(game._board, game._player)
    assert not _get_legal_moves(game._board, game._player * -1)
    assert game.game_over
    assert game.running_tile_count == 13

    assert game.tile_count[1] == 9
    assert game.tile_count[-1] == 4


def test_no_legal_moves() -> None:
    board = np.array(
        [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, -1.0], [1.0, 1.0, 1.0, 0.0], [0.0, -1.0, -1.0, -1.0]]
    )
    assert not _get_legal_moves(board, 1)
    assert not _get_legal_moves(board, -1)


def test_is_legal_move_one_player_only() -> None:

    board = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    move = (7, 7)
    current_player = -1
    assert not is_legal_move(board, move, current_player)

    current_player = 1
    assert is_legal_move(board, move, current_player)

    # def test_has_flip() -> None:
    # """ part of failed speedup attempt"""


#     line = np.array([0, 0, 0, 1, -1, -1, 2]).astype(float)
#     assert has_tile_to_flip(line)

#     line = np.array([0, 1, -1, 1, -1, 1, 2]).astype(float)
#     assert not has_tile_to_flip(line)


def test_illegal_move() -> None:
    with pytest.raises(AssertionError):
        game = OthelloEnv()
        game.reset()
        illegal_move = (int(game.board_dim / 2), int(game.board_dim / 2))
        game.step(illegal_move)


def test_jacob_board() -> None:
    board = np.array(
        [
            ["O", "X", "*", "*", "*", "*", "*", "*"],
            ["X", "X", "X", "*", "*", "*", "*", "*"],
            ["*", "X", "O", "X", "*", "*", "*", "*"],
            ["*", "*", "X", "O", "X", "*", "*", "*"],
            ["*", "*", "*", "X", "O", "X", "*", "*"],
            ["*", "*", "*", "*", "X", "X", "X", "*"],
            ["*", "*", "*", "*", "*", "*", "O", "*"],
            ["*", "*", "*", "*", "*", "*", "*", "O"],
        ]
    )
    board[board == "O"] = "2"
    board[board == "X"] = "1"
    board[board == "*"] = "0"
    board = board.astype(float)
    # Can't use "-1" as string, so dumb workaround
    board[board == 2] = -1
    assert has_legal_move(board, current_player=1)
    assert has_legal_move(board, current_player=-1)


# def test_idx_surrounding() -> None:
#     """idx_surrouding is part of failed speed up attempt"""
#     pass

# board = np.array(
#     [
#         ["O", "X", "*", "*", "*", "*", "*", "*"],
#         ["X", "X", "X", "*", "*", "*", "*", "*"],
#         ["*", "X", "O", "X", "*", "*", "*", "*"],
#         ["*", "*", "X", "O", "X", "*", "*", "*"],
#         ["*", "*", "*", "X", "O", "X", "*", "*"],
#         ["*", "*", "*", "*", "X", "X", "X", "*"],
#         ["*", "*", "*", "*", "*", "*", "O", "*"],
#         ["*", "*", "*", "*", "*", "*", "*", "O"],
#     ]
# )
# board[board == "O"] = "2"
# board[board == "X"] = "1"
# board[board == "*"] = "0"
# board = board.astype(float)
# # Can't use "-1" as string, so dumb workaround
# board[board == 2] = -1
# idxs = idx_surrounding(board, (6, 6))
# assert (7, 6) in idxs
