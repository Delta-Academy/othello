import random
from typing import Optional, Tuple

import numpy as np
import pytest

from game_mechanics import OthelloEnv, _get_legal_moves, has_flip, is_legal_move


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
    expected = np.array(
        [
            [0] * 4,
            [0, -1, 1, 0],
            [0, 1, -1, 0],
            [0] * 4,
        ],
        dtype=float,
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
    ).astype(float)
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
    ).astype(float)
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
    ).astype(float)
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
    ).astype(float)
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


def test_has_flip() -> None:
    line = np.array([0, 0, 0, 1, -1, -1, 2]).astype(float)
    assert has_flip(line)

    line = np.array([0, 1, -1, 1, -1, 1, 2]).astype(float)
    assert not has_flip(line)


def test_illegal_move() -> None:
    with pytest.raises(AssertionError):
        game = OthelloEnv()
        illegal_move = (int(game.board_dim / 2), int(game.board_dim / 2))
        game.step(illegal_move)
