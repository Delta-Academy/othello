import copy
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numba as nb
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).parent.resolve()


def load_network(team_name: str) -> nn.Module:
    net_path = HERE / f"{team_name}_network.pt"
    assert (
        net_path.exists()
    ), f"Network saved using TEAM_NAME='{team_name}' doesn't exist! ({net_path})"
    model = torch.load(net_path)
    model.eval()
    return model


def save_network(network: nn.Module, team_name: str) -> None:
    assert isinstance(
        network, nn.Module
    ), f"train() function outputs an network type: {type(network)}"
    assert "/" not in team_name, "Invalid TEAM_NAME. '/' are illegal in TEAM_NAME"
    net_path = HERE / f"{team_name}_network.pt"
    n_retries = 5
    for attempt in range(n_retries):
        try:
            torch.save(network, net_path)
            load_network(team_name)
            return
        except Exception:
            if attempt == n_retries - 1:
                raise


########## POTENTIALLY USEFUL FEATURES ############################


def get_legal_moves(board: np.ndarray) -> List[Tuple[int, int]]:
    """Return a list of legal moves that can be made by player 1 on the current board."""
    return _get_legal_moves(board, 1)


def make_move(board: np.ndarray, move: Tuple[int, int]) -> np.ndarray:
    """Returns board after move has been made by player 1."""
    return _make_move(board, move, 1)


def choose_move_randomly(
    board: np.ndarray,
) -> Optional[Tuple[int, int]]:
    """Returns a random legal move on the current board (always plays as player 1)."""
    moves = get_legal_moves(board)
    if moves:
        # Fast rng using microsecond digit of time
        # gives uniform distribution 0-9
        # (len(moves) is rarely > 10)
        idx = int(time.time()) % 10 % len(moves)
        return moves[idx]
    return None


####### THESE FUNCTIONS ARE LESS USEFUL ############


def _make_move(board: np.ndarray, move: Tuple[int, int], current_player: int) -> np.ndarray:
    """Returns board after move has been made."""
    if is_legal_move(board, move, current_player):
        board_after_move = copy.deepcopy(board)
        board_after_move[move] = current_player
        board_after_move = flip_tiles(board_after_move, move, current_player)
    else:
        raise ValueError(f"Move {move} is not a valid move!")
    return board_after_move


def is_legal_move(board: np.ndarray, move: Tuple[int, int], current_player: int) -> bool:
    board_dim = board.shape[0]
    if is_valid_coord(board_dim, move[0], move[1]) and board[move] == 0:
        if has_tile_to_flip(board, move, current_player):
            return True
    return False


def idx_surrounding(arr: np.ndarray, idx: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the indexes of the elements within a radius of 1 of the original element (arr must be
    2d and square currently)"""
    assert arr.shape[0] == arr.shape[1]
    dim = arr.shape[0]
    x, y = idx
    for num in (x, y):
        assert 0 <= num < dim
    x_idx = np.arange(x - 1, x + 2)
    y_idx = np.arange(y - 1, y + 2)

    # Remove out of array indexes
    x_idx = x_idx[np.logical_and(x_idx >= 0, x_idx < dim)]
    y_idx = y_idx[np.logical_and(y_idx >= 0, y_idx < dim)]

    return x_idx, y_idx


def is_valid_coord(board_dim: int, row: int, col: int) -> bool:
    return 0 <= row < board_dim and 0 <= col < board_dim


def get_vectors(
    arr: np.ndarray, move: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get all 4 vectors in arr that pass through coordinate move."""
    dim = arr.shape[0]
    vertical = arr[np.arange(dim), move[1]]
    horizontal = arr[move[0], np.arange(dim)]

    # Top left to bottom right diagonal
    east = dim - max(move)
    west = min(move)
    row = np.arange(move[0] - west, move[0] + east)
    col = np.arange(move[1] - west, move[1] + east)
    diag1 = arr[row, col]

    # Bottom left to top right diagonal
    west = min(dim - 1 - move[0], move[1])
    east = min(move[0], dim - 1 - move[1])
    row = np.flip(np.arange(move[0] - east, move[0] + west + 1))
    col = np.arange(move[1] - west, move[1] + east + 1)
    diag2 = arr[row, col]

    return vertical, horizontal, diag1, diag2


@nb.jit(parallel=True)
def is_sub_arr(a1: np.ndarray, a2: np.ndarray) -> bool:
    """is a2 a subarray of a1?"""
    for i in nb.prange(len(a1) - len(a2) + 1):
        for j in range(len(a2)):
            if a1[i + j] != a2[j]:
                break
        else:
            return True
    return False


# Current only works for dim<=8 and not optimised for smaller
MATCHES = [
    np.array([2, -1, 1], dtype=float),
    np.array([2, -1, -1, 1], dtype=float),
    np.array([2, -1, -1, -1, 1], dtype=float),
    np.array([2, -1, -1, -1, -1, 1], dtype=float),
    np.array([2, -1, -1, -1, -1, -1, 1], dtype=float),
    np.array([2, -1, -1, -1, -1, -1, -1, 1], dtype=float),
    np.array([1, -1, 2], dtype=float),
    np.array([1, -1, -1, 2], dtype=float),
    np.array([1, -1, -1, -1, 2], dtype=float),
    np.array([1, -1, -1, -1, -1, 2], dtype=float),
    np.array([1, -1, -1, -1, -1, -1, 2], dtype=float),
    np.array([1, -1, -1, -1, -1, -1, -1, 2], dtype=float),
]


def has_flip(line: np.ndarray) -> bool:
    for match in MATCHES:
        if is_sub_arr(line, match):
            return True
    return False


def has_tile_to_flip(
    board: np.ndarray,
    move: Tuple[int, int],
    current_player: int,
) -> bool:
    """Checks if there is a tile to flip in any direction following move."""
    board_check = board.copy()
    # Mark the potential move as a 2, to distinguish from existing counters
    board_check[move] = 2 * current_player
    for vector in get_vectors(board_check * current_player, move):
        if has_flip(vector):
            return True
    return False


def has_tile_to_flip_direction(
    board: np.ndarray,
    move: Tuple[int, int],
    direction: Tuple[int, int],
    current_player: int,
) -> bool:
    """Following move, check if there is a tile to flip in a given direction.

    n.b. This is not optimised but is called rarely so not 100% necessary
    """
    board_dim = board.shape[0]
    i = 1
    while True:
        row = move[0] + direction[0] * i
        col = move[1] + direction[1] * i
        if not is_valid_coord(board_dim, row, col) or board[row, col] == 0:
            return False
        elif board[row, col] == current_player:
            break
        else:
            i += 1

    return i > 1


def flip_tiles(board: np.ndarray, move: Tuple[int, int], current_player: int) -> np.ndarray:
    """Flips the adversary's tiles for current move and updates the running tile count for each
    player.

    Arg:
        move: The move just made to
               trigger the flips
    """
    for direction in MOVE_DIRS:
        if has_tile_to_flip_direction(board, move, direction, current_player):
            i = 1
            while True:
                row = move[0] + direction[0] * i
                col = move[1] + direction[1] * i
                if board[row][col] == current_player:
                    break
                else:
                    board[row][col] = current_player
                    i += 1
    return board


def get_possible_moves(board: np.ndarray) -> set:

    board_dim = len(board)
    zero_idx = board == 0
    # Few empty, iterate through these (1.5 threshold chosen empirically for speed)
    if np.sum(zero_idx) < board_dim**2 / 1.5:
        idx_check = np.where(zero_idx)
        to_check = set(zip(idx_check[0], idx_check[1]))
    else:
        to_check = set()
        idx_pieces = np.where(~zero_idx)
        for pos in zip(idx_pieces[0], idx_pieces[1]):
            idxs = idx_surrounding(board, pos)
            to_check.update(list(zip(idxs[0], idxs[1])))

    return to_check


def has_legal_move(board: np.ndarray, current_player: int) -> bool:
    """Checks whether current_player has any legal move to make."""
    possible_moves = get_possible_moves(board)
    for move in possible_moves:
        if is_legal_move(board, move, current_player):
            return True
    return False


def _get_legal_moves(board: np.ndarray, current_player: int) -> List[Tuple[int, int]]:
    possible_moves = get_possible_moves(board)
    return [move for move in possible_moves if is_legal_move(board, move, current_player)]


def get_empty_board(board_dim: int = 6, player_start: int = 1) -> np.ndarray:
    board = np.zeros((board_dim, board_dim))
    if board_dim < 2:
        return board
    coord1 = int(board_dim / 2 - 1)
    coord2 = int(board_dim / 2)
    initial_squares = [
        (coord1, coord2),
        (coord1, coord1),
        (coord2, coord1),
        (coord2, coord2),
    ]

    for i in range(len(initial_squares)):
        row = initial_squares[i][0]
        col = initial_squares[i][1]
        player = player_start if i % 2 == 0 else player_start * -1
        board[row, col] = player

    return board.copy()


# Directions relative to current counter (0, 0) that a tile to flip can be
MOVE_DIRS = [(-1, -1), (-1, 0), (-1, +1), (0, -1), (0, +1), (+1, -1), (+1, 0), (+1, +1)]


class OthelloEnv:
    # Constants for rendering
    DISC_SIZE_RATIO = 0.8
    SQUARE_SIZE = 60

    BLUE_COLOR = (23, 93, 222)
    BACKGROUND_COLOR = (19, 72, 162)
    BLACK_COLOR = (0, 0, 0)
    WHITE_COLOR = (255, 255, 255)
    YELLOW_COLOR = (255, 240, 0)
    RED_COLOR = (255, 0, 0)

    def __init__(
        self,
        opponent_choose_move: Callable[
            [np.ndarray], Optional[Tuple[int, int]]
        ] = choose_move_randomly,
        board_dim: int = 8,
    ):
        self._board_visualizer = np.vectorize(lambda x: "X" if x == 1 else "O" if x == -1 else "*")
        self._opponent_choose_move = opponent_choose_move
        self._screen = None
        self.board_dim = board_dim
        self.reset()

    def reset(self, verbose: bool = False) -> Tuple[np.ndarray, int, bool, Dict]:
        """Resets game & takes 1st opponent move if they are chosen to go first."""
        self._player = random.choice([-1, 1])
        self._board = get_empty_board(self.board_dim, self._player)
        self.running_tile_count = 4 if self.board_dim > 2 else 0
        self.done = False
        self.winner = None
        if verbose:
            print(f"Starting game. Player {self._player} has first move\n", self)

        reward = 0

        if self._player == -1:
            # Negative sign is because both players should see themselves as player 1
            opponent_action = self._opponent_choose_move(-self._board)
            reward = -self._step(opponent_action, verbose)  # Can be None, fix

        return self._board, reward, self.done, self.info

    def __repr__(self) -> str:
        return str(self._board_visualizer(self._board)) + "\n"

    @property
    def info(self) -> Dict[str, Any]:
        return {"player_to_take_next_move": self._player, "winner": self.winner}

    @property
    def tile_count(self) -> Dict:
        return {1: np.sum(self._board == 1), -1: np.sum(self._board == -1)}

    def switch_player(self) -> None:
        """Change self.player only when game isn't over."""
        self._player *= -1 if not self.done else 1

    def _step(self, move: Optional[Tuple[int, int]], verbose: bool = False) -> int:
        """Takes 1 turn, internal to this class.

        Do not call
        """
        assert not self.done, "Game is over, call .reset() to start a new game"

        if move is None:
            assert not has_legal_move(
                self._board, self._player
            ), f"Your move is None, but you must make a move when a legal move is available!"
            if verbose:
                print(f"Player {self._player} has no legal move, switching player")
            self.switch_player()
            return 0

        assert is_legal_move(self._board, move, self._player), f"Move {move} is not valid!"

        self.running_tile_count += 1
        self._board = _make_move(self._board, move, self._player)

        # Check for game completion
        tile_difference = self.tile_count[self._player] - self.tile_count[self._player * -1]
        self.done = self.game_over
        self.winner = (
            None
            if self.tile_count[1] == self.tile_count[-1] or not self.done
            # mypy sad, probably bug: github.com/python/mypy/issues/9765
            else max(self.tile_count, key=self.tile_count.get)  # type: ignore
        )
        won = self.done and tile_difference > 0

        # Currently just if won, many alternatives
        reward = 1 if won else 0

        if verbose:
            print(f"Player {self._player} places counter at row {move[0]}, column {move[1]}")
            print(self)
            if self.done:
                if won:
                    print(f"Player {self._player} has won!\n")
                elif self.running_tile_count == self.board_dim**2 and tile_difference == 0:
                    print("Board full. It's a tie!")
                else:
                    print(f"Player {self._player * -1} has won!\n")

        self.switch_player()
        return reward

    def step(
        self, move: Optional[Tuple[int, int]], verbose: bool = False
    ) -> Tuple[np.ndarray, int, bool, Dict[str, int]]:
        """Called by user - takes 2 turns, yours and your opponent's"""

        reward = self._step(move, verbose)

        if not self.done:
            # Negative sign is because both players should see themselves as player 1
            opponent_action = self._opponent_choose_move(-self._board)
            opponent_reward = self._step(opponent_action, verbose)  # Can be None, fix
            # Negative sign is because the opponent's victory is your loss
            reward -= opponent_reward

        if self.done:
            if np.sum(self._board == 1) > np.sum(self._board == -1):
                reward = 1
            elif np.sum(self._board == 1) < np.sum(self._board == -1):
                reward = -1
            else:
                reward = 0
        else:
            reward = 0

        return self._board, reward, self.done, self.info

    @property
    def game_over(self) -> bool:
        if self.running_tile_count == self.board_dim**2:
            return True
        elif has_legal_move(self._board, self._player):
            return False
        elif has_legal_move(self._board, self._player * -1):
            return False
        return True

    ## TODO: Pygame stuff
    # def __del__(self):
    #     """Destructor, quit pygame if game over."""
    #     if self._screen is not None:
    #         pygame.quit()

    # def render(self) -> None:
    #     """Renders game in pygame."""
    #     if self._screen is None:
    #         pygame.init()
    #         self._screen = pygame.display.set_mode(
    #             (self.SQUARE_SIZE * self.N_COLS, self.SQUARE_SIZE * self.N_ROWS)
    #         )

    #     # Draw background of the board
    #     pygame.gfxdraw.box(
    #         self._screen,
    #         pygame.Rect(
    #             0,
    #             0,
    #             self.N_COLS * self.SQUARE_SIZE,
    #             self.N_ROWS * self.SQUARE_SIZE,
    #         ),
    #         self.BLUE_COLOR,
    #     )

    #     # Draw the circles - either as spaces if filled or
    #     for r in range(self.N_ROWS):
    #         for c in range(self.N_COLS):
    #             space = self._board[r, c]
    #             colour = (
    #                 self.RED_COLOR
    #                 if space == 1
    #                 else self.YELLOW_COLOR
    #                 if space == -1
    #                 else self.BACKGROUND_COLOR
    #             )

    #             # Anti-aliased circle drawing
    #             pygame.gfxdraw.aacircle(
    #                 self._screen,
    #                 c * self.SQUARE_SIZE + self.SQUARE_SIZE // 2,
    #                 r * self.SQUARE_SIZE + self.SQUARE_SIZE // 2,
    #                 int(self.DISC_SIZE_RATIO * self.SQUARE_SIZE / 2),
    #                 colour,
    #             )

    #             pygame.gfxdraw.filled_circle(
    #                 self._screen,
    #                 c * self.SQUARE_SIZE + self.SQUARE_SIZE // 2,
    #                 r * self.SQUARE_SIZE + self.SQUARE_SIZE // 2,
    #                 int(self.DISC_SIZE_RATIO * self.SQUARE_SIZE / 2),
    #                 colour,
    #             )
    #     pygame.display.update()


def play_othello_game(
    your_choose_move: Callable[[np.ndarray], Optional[Tuple[int, int]]],
    opponent_choose_move: Callable[[np.ndarray], Optional[Tuple[int, int]]],
    game_speed_multiplier: float = 1,
    render: bool = False,
    verbose: bool = False,
) -> int:
    """Play a game where moves are chosen by `your_choose_move()` and `opponent_choose_move()`. Who
    goes first is chosen at random. You can render the game by setting `render=True`.

    Args:
        your_choose_move: function that chooses move (takes state as input)
        opponent_choose_move: function that picks your opponent's next move
        game_speed_multiplier: multiplies the speed of the game. High == fast
        render: whether to render the game using pygame or not
        verbose: whether to print board states to console. Useful for debugging

    Returns: total_return, which is the sum of return from the game
    """
    total_return = 0
    game = OthelloEnv(opponent_choose_move)
    state, reward, done, info = game.reset(verbose)
    # if render:
    #     game.render()
    if verbose:
        time.sleep(1 / game_speed_multiplier)

    while not done:
        action = your_choose_move(state)
        state, reward, done, info = game.step(action, verbose)
        # if render:
        #     game.render()
        total_return += reward
        if verbose:
            time.sleep(1 / game_speed_multiplier)

    return total_return
