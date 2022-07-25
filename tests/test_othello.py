import random
from typing import Optional, Tuple

import numpy as np

import pytest
from game_mechanics import (
    OthelloEnv,
    _get_legal_moves,
    choose_move_randomly,
)


def choose_move4x4(state) -> Optional[Tuple[int, int]]:
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


# def test_8x8() -> None:
#     game = OthelloEnv(opponent_choose_move=choose_move4x4, board_dim=4)

#     expected = np.array(
#         [
#             [0] * 8,
#             [0] * 8,
#             [0] * 8,
#             [0, 0, 0, -1, 1, 0, 0, 0],
#             [0, 0, 0, 1, -1, 0, 0, 0],
#             [0] * 8,
#             [0] * 8,
#             [0] * 8,
#         ]
#     )
#     np.testing.assert_array_equal(game._board, expected)

#     expected = np.array(
#         [
#             [0] * 8,
#             [0] * 8,
#             [0] * 8,
#             [0, 0, 1, 1, 1, 0, 0, 0],
#             [0, 0, 0, 1, -1, 0, 0, 0],
#             [0] * 8,
#             [0] * 8,
#             [0] * 8,
#         ]
#     )
#     np.testing.assert_array_equal(game._board, expected)

#     expected = np.array(
#         [
#             [0] * 8,
#             [0] * 8,
#             [0, 0, 0, 0, -1, 0, 0, 0],
#             [0, 0, 1, 1, -1, 0, 0, 0],
#             [0, 0, 0, 1, -1, 0, 0, 0],
#             [0] * 8,
#             [0] * 8,
#             [0] * 8,
#         ]
#     )
#     np.testing.assert_array_equal(game._board, expected)

#     expected = np.array(
#         [
#             [0] * 8,
#             [0, 0, 0, 0, 0, 1, 0, 0],
#             [0, 0, 0, 0, 1, 0, 0, 0],
#             [0, 0, 1, 1, -1, 0, 0, 0],
#             [0, 0, 0, 1, -1, 0, 0, 0],
#             [0] * 8,
#             [0] * 8,
#             [0] * 8,
#         ]
#     )
#     np.testing.assert_array_equal(game._board, expected)
#     while not game.game_over:
#         game.make_random_move()
#         print(game.running_tile_count)
#         print(game._board)
#     assert game.running_tile_count > 58  # Arbitrary, probably true
#     assert game.running_tile_count <= 64


def test_illegal_move() -> None:
    with pytest.raises(AssertionError):
        game = OthelloEnv()
        illegal_move = (int(game.board_dim / 2), int(game.board_dim / 2))
        game.step(illegal_move)
