##  RL to play Othello :black_circle::white_circle::black_circle::white_circle:


![Othello. The opening state of the board.](./images/Othello-Standard-Board.jpeg)

## Rules of Othello :black_circle:

Othello is a **two-player** board game. Players take turns placing counters onto a  grid. Each player **plays a single color** only. The pieces **must be placed into a square on the grid**. 

The game starts with 4 pieces already placed, 2 from each side, sitting diagonally from each other in the centre of the board.  
:black_circle: :white_circle:  
:white_circle: :black_circle:

Subsequent pieces must always be placed so that the player has **outflanked at least one** opponent piece.

This means that the player must place a piece on the board so that there exists at least one straight (horizontal, vertical, or diagonal) occupied line between the new piece and another of the player's pieces, with one or more contiguous opponent pieces between them. 

This is demonstrated below. Black can only place a piece on the grey squares, as these are the only squares that **outflank** and trap a white piece between black pieces. 



![Valid moves for Black shown in Grey.](./images/validMovesForBlack.png)


After a player places a piece, **outflanked** opponent pieces are flipped so that they become the colour of the player playing the **outflanking** move.

![Before: white can play in grey squares. After: white flips a black counter.](./images/beforeAfterMoveWhite.png)

**The goal is have more counters of your color on the board than your opponent at the end of the game.**

The game continues until neither player has a valid move. Meaning no outflanking move can be played, or all 64 squares are occupied.

If a player has no valid move, but their opponent does, the turn is switched to the opponent without a piece being played.

You can play with your teammate here: https://www.eothello.com/.

Your task is to build a **Reinforcement Learning agent** that plays **Othello**.

# Competition Rules :scroll:
1. You must build a **Reinforcement Learning** agent.
    - Rules-based agents aren't allowed!
2. You can only write code in `main.py` and you can only store data to be used as a `torch` neural network (`save_network()` saves in a `.pt` file)
    - In the competition, your agent will call the `choose_move()` function in `main.py` to select a move (`choose_move()` may call other functions in `main.py`)
    - Any code not in `main.py` will not be used.    
3. Submission deadline: **4pm GMT, Sunday**.
    - You can update your code after submitting, but **not after the deadline**.
    - Check your submission is valid with `check_submission()`


## Competition Format :crossed_swords:

The competition is a knockout tournament where your AI will play other teams' AIs 1-v-1.

We follow the Othello World Championship Rules. Each 1-v-1 matchup consists of up to 3 games. The **first player to play** the first game is chosen randomly. The other player starts the second game. The starter of the third game, if one is necessary, is the player who, in total over games 1 & 2, had the most discs. If this is a tie, the draw is made randomly.

The winner of the matchup is the winner of more of the 3 games. If this is tied, the total number of discs at the end of the games shall be used to determine the winner. If this is a tie, then both teams progress to the next round with one of the teams as their representative.

E.g. if a player wins 2 games, they win overall.

If both players win 1, draw 1 and lose 1, then the difference in the number of pieces between the win and the loss decides the matchup (since the draw has an equal number of pieces).

The competition & discussion will be in [Gather Town](https://app.gather.town/app/nJwquzJjD4TLKcTy/Delta%20Academy) at **5pm GMT on Sunday** (60 mins after submission deadline)!

![Example knockout tournament tree](./images/tournament_tree.png)

## Technical Details :hammer:

### States :white_circle:
Othello is played on a **8 x 8** board.

The **first axis is the row index** and the **2nd axis is the column index**.

In the code, the state is represented as a `numpy` array. The pieces are integers in this array. An empty space is `0`. Your pieces are denoted `1`. Your opponent's pieces are denoted `-1`.

Since there are `10 ** 28` possible states, we suggest you use a neural network or other representational method to reduce the state space. 

### Actions :axe:

**The index (0 -> 7) of the column and the index (0 -> 7) of the row of the space to place your counter - as a Tuple `(row, column)`.** 

In Othello, sometimes you will have no valid move to take but the game is not finished. In this case, you must `return None`.

If a legal move is available you must play it. So only `return None` when no legal move is available. 


### Rewards :moneybag:

You receive `+1` for winning, `-1` for losing and `0` for a draw. You receive `0` for all other moves.



## Functions you write :point_left:

<details>
<summary><code style="white-space:nowrap;">  train()</code></summary>
Write this to train your network from experience in the environment. 
<br />
<br />
Return the trained network so it can be saved.
</details>
<details>
<summary><code style="white-space:nowrap;">  choose_move()</code></summary>
This acts greedily given the state and network.

In the competition, the choose_move() function is called to make your next move. Takes the state as input and outputs an action.

Also has a verbose mode, which when set to True prints to console the legal moves. Useful for debugging.

</details>

## Existing Code :pray:

### Need to Know

<details>
<summary><code style="white-space:nowrap;">OthelloEnv</code> class</summary>
The environment class controls the game and runs the opponent. It should be used for training your agent.
<br />
<br />
See example usage in <code style="white-space:nowrap;">play_othello_game()</code>.
<br />
<br />
The opponent's <code style="white-space:nowrap;">choose_move</code> function is input at initialisation (when <code style="white-space:nowrap;">Env(opponent_choose_move)</code> is called). The first player is chosen at random when <code style="white-space:nowrap;">Env.reset()</code> is called. Every time you call <code style="white-space:nowrap;">Env.step()</code>, 2 moves are taken - yours and then your opponent's. Your opponent sees a 'flipped' version of the board, where his pieces are shown as <code style="white-space:nowrap;">1</code>'s and yours are shown as <code style="white-space:nowrap;">-1</code>'s.
    <br />
    <br />
    Both <code style="white-space:nowrap;">  Env.step()</code> and <code style="white-space:nowrap;">  Env.reset()</code> have <code style="white-space:nowrap;">  verbose</code> arguments which print debugging info to console when set to <code style="white-space:nowrap;">True</code>. Verbose visualises the ongoing progress of the game in the console. Player1's tiles (you) are represented as an X. Your opponents tiles are represented as O.
</details>


<details>
<summary><code style="white-space:nowrap;">  choose_move_randomly()</code></summary>
Like above, but randomly picks from available legal moves.
<br />
<br />
Takes the state as input and outputs an action.
</details>

<details>
<summary><code style="white-space:nowrap;">  play_othello_game()</code></summary>
Plays 1 game of Othello, which can be visualsed in the console (if <code style="white-space:nowrap;">verbose=True</code>) . Outputs the return for your agent.
<br />
<br />
Inputs:
    
<code style="white-space:nowrap;">your_choose_move</code>: Function that takes the state and outputs the action for your agent.

<code style="white-space:nowrap;">opponent_choose_move</code>: Function that takes the state and outputs the action for the opponent.

<code style="white-space:nowrap;">game_speed_multiplier</code>: controls the gameplay speed. High numbers mean fast games, low numbers mean slow games.


<code style="white-space:nowrap;">verbose</code>: whether to print to console each move and the corresponding board states.
</details>


## Other code :gear:

There are **a load** of functions in `game_mechanics.py`. The useful functions are clearly indicated and are explained in their docstrings. **Feel free to use these, but don't change them.** This is because the original `game_mechanics.py` file will be used in the competition.

If you want to tweak one, copy-paste it to `main.py` and rename it.

## Suggested Approach :+1:

1. Discuss your neural network architecture - how many inputs, outputs, hidden layers & which activation functions should you use
2. **Write `train()`** (you can borrow code from past exercises).
3. Insert debugging messages - you want to make sure that:
   - Loss is decreasing :chart_with_downwards_trend:
   - The magnitude of update steps are decreasing :arrow_down:
   - Performance on Othello is improving :arrow_up:
4. Iterate on the neural network architecture, hyperparameters & training algorithm
