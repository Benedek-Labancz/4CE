import numpy as np
import math

# Numeric representation of symbols
X = 1
O = -1
E = 0
M = 2

# str symbols
str_symbols = {
    -1: "O",
    1: "X",
    0: "_",
    2: "M"
}




def print_state_old(state):
    state_view = state.reshape(-1)
    state_view = list(state_view)
    state_str = ""
    for i in range(len(state_view)):
        if (i + 1) % 3 == 0:
            state_str += ' ' + str_symbols[state_view[i]] + ' '
        else:
            state_str += ' ' + str_symbols[state_view[i]]
        if (i + 1) % 9 == 0:
            state_str += '\n'
        if (i + 1) % 27 == 0:
            state_str += '\n'
    print(state_str)

def print_state(state):
    state_str = ""
    K, L, x, y = 0, 0, 0, 0
    for i in range(81):
        K = math.floor(i / 3) % 3
        L = math.floor(i / 27)
        x = i % 3
        y = math.floor(i / 9) % 3
        tile = str_symbols[state[K, L, x, y]]
        state_str += ' ' + tile
        if x == 2: state_str += ' '
        if K == 2 and x == 2: state_str += '\n'
        if K == 2 and x == 2 and y == 2: state_str += '\n'
    print(state_str)

def init_state():
    state = np.zeros((3, 3, 3, 3))
    state[1, 1, 1, 1] = M # middle tile
    return state

def perform_action(state, action, player):
    '''
    Write the player's symbol to the tile defined by action.
    '''
    state[action] = player
    return state

def valid_action(state, action):
    if type(action) != tuple:
        return False
    for i in range(len(action)):
        try:
            if action[i] < 0 or action[i] > 2:
                return False
        except:
            return False
    try:
        if state[action] != E:
            return False
    except:
        return False
    return True

def calculate_scoring_positions():
    '''
    Calculate all the possible scoring positions in the form of {(K, L, x, y), (K, L, x, y), (K, L, x, y)}
    '''
    sc = {} # set of all possible scoring positions
    # 2 dimensions
    for K in [0, 1, 2]:
        for L in [0, 1, 2]:
            pass

def evaluate_action(state, action):
    '''
    Count the number of points scored by an action.

    Args:
        state: the state following the action
        action: the action taken by the actor

    I have chosen to include the state following the action in order to not rely on
    having to pass or calculate the player.
    '''

    player = state[action]
    clean_state = (state == player)


def terminal_state(state):
    '''
    Returns True if game state is terminal,
    i.e. there are no empty tile left.
    '''
    return (state == E).sum() == 0

class Game:
    def __init__(self):
        self.state = init_state()
        self.player = X
        self.X_score = 0
        self.O_score = 0

    def switch_turn(self):
        if self.player == X:
            self.player = O
        else:
            self.player = X


if __name__ == '__main__':

    print("Welcome to 4CE!")
    game = Game()
    print_state(game.state)
    while True:
        
        action = None
        while not valid_action(game.state, action):
            inp = input(f"Player {str_symbols[game.player]}'s turn: ")
            action = tuple([int(coord) for coord in inp.split(' ')])

        game.state = perform_action(game.state, action, game.player)
        print_state(game.state)
        
        if terminal_state(game.state):
            print("The game has ended.")
            print(f"Final score: {str_symbols[X]}: {game.X_score}  {str_symbols[O]}: {game.O_score}")
            break
        else:
            game.switch_turn()