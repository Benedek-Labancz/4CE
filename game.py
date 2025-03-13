import torch
import numpy as np
import math
from copy import deepcopy

# Numeric and string representation of symbols; needed for easy refactoring
X = 1
O = -1
E = 0
M = 2

str_symbols = {
    O: "O",
    X: "X",
    E: "_",
    M: "M"
}

def print_state(state):
    '''
    Prints a state in the terminal.
    Terminal has to be cleared separately for elegance (see example usage).
    '''
    state_str = ""
    K, L, x, y = 0, 0, 0, 0
    for i in range(81):
        K = math.floor(i / 3) % 3
        L = math.floor(i / 27)
        x = i % 3
        y = math.floor(i / 9) % 3
        tile = str_symbols[state[K, L, x, y].item()]
        state_str += ' ' + tile
        if x == 2: state_str += ' '
        if K == 2 and x == 2: state_str += '\n'
        if K == 2 and x == 2 and y == 2: state_str += '\n'
    print(state_str)

def print_score(score):
    '''
    Prints game score in the terminal.
    '''
    print(f"[X - {score[X]}\tO - {score[O]}]")

class Game:
    def __init__(self):
        self.state = self.init_state()
        self.player = X
        self.score = {
            X: 0,
            O: 0
        }
        self.action_spec = { # the action specification of the game
            "shape": (3, 3, 3, 3),
            "n": 81 # number of all hypothetical actions (middle is always invalid)
        }

        self.scoring_positions = self.calculate_scoring_positions()

    def step(self, action):
        '''
        Computes the next state of the environment along with the reward
        given the current state of the game and an action.

        The function is designed to look similar to the OpenAI Gym API,
        hence the form of the return is `new_state, reward, done, info`,
        where done indicates whether new_state is terminal, and info is ignored for now.
        Note that the action is not directly preformed, therefore the state of the game
        has to be overwritten manually (see example usage).
        '''
        # validate
        if not self.valid_action(self.state, action):
            raise Exception("Invalid action encountered")

        # stepping the env
        new_state = self.perform_action(deepcopy(self.state), action, self.player)

        # bookkeeping
        points_scored = self.evaluate_action(new_state, action)
        self.score[self.player] += points_scored

        reward = points_scored
        done = self.terminal_state(new_state)
        info = '' # TODO

        return new_state, reward, done, info

    def switch_turn(self):
        if self.player == X:
            self.player = O
        else:
            self.player = X

    def calculate_scoring_positions(self):
        '''
        Calculate all the possible scoring positions in the form of {(K, L, x, y), (K, L, x, y), (K, L, x, y)}
        '''
        '''
        vertical
        {(0, 0), (0, 1), (0, 2)}
        {(1, 0), (1, 1), (1, 2)}
        {(2, 0), (2, 1), (2, 2)}
        
        horizontal
        {(0, 0), (1, 0), (2, 0)}
        {(0, 1), (1, 1), (2, 1)}
        {(0, 2), (1, 2), (2, 2)}

        diagonal
        {(0, 0), (1, 1), (2, 2)}
        {(0, 2), (1, 1), (2, 0)}
        '''
        '''
        column (9) same x, y
        face-diag (8)
        body-diag (4)
        '''
        '''
        Total number of positions: 72 (2d) + 72 (columns) + 2x8x8(body and face diag) = 272
        '''
        def desc(x, y):
            '''
            Descartes product of two tuples.
            e.g.
            x = (0)
            y = (0, 1, 2)
            return [(0, 0), (0, 1), (0, 2)]
            '''  
            fin = []
            for i in range(len(x)):
                for j in range(len(y)):
                    fin.append((x[i], y[j]))
            return fin

        sc = [] # set of all possible scoring positions
        horizontal_xy = [tuple(desc((0, 1, 2), (i,))) for i in [0, 1, 2]]
        vertical_xy = [tuple(desc((i,), (0, 1, 2))) for i in [0, 1, 2]]
        diagonal_xy = [((0, 0), (1, 1), (2, 2)), ((0, 2), (1, 1), (2, 0))]
        xy_options = horizontal_xy + vertical_xy + diagonal_xy # all the possible lineups in 2 dimensions

        # 2 dimensions
        for K in [0, 1, 2]:
            for L in [0, 1, 2]:
                for xy_trio in xy_options:
                    sc.append([(K, L) + xy for xy in xy_trio])

        # face and body diagonals
        for kl_trio in xy_options:
            for xy_trio in xy_options:
                regular = [kl_trio[i] + xy_trio[i] for i in [0, 1, 2]]
                reverse = [kl_trio[i] + xy_trio[2 - i] for i in [0, 1, 2]]
                sc += [regular, reverse]
        
        #columns
        for kl_trio in xy_options:
            for x in [0, 1, 2]:
                for y in [0, 1, 2]:
                    sc.append([kl + (x, y) for kl in kl_trio])   
        return sc

    def init_state(self):
        '''
        Initialize game state, i.e. a blank board.
        '''
        state = torch.zeros((3, 3, 3, 3))
        state[1, 1, 1, 1] = M # middle tile
        return state

    def first_person_state(self, state):
        '''
        Convert a state to first-person.
        A first-person state's symbols are overwritten to X for the current player
        and O for the opponent.
        '''
        def write_symbols(state, coords, symbol):
            for coord in coords:
                state[tuple(coord)] = symbol
            return state

        me_coords = np.transpose((np.array(state) == self.player).nonzero())
        other_player = O if self.player == X else X
        you_coords = np.transpose((np.array(state) == other_player).nonzero())
        blank_state = self.init_state()
        fp_state = write_symbols(blank_state, me_coords, X)
        fp_state = write_symbols(fp_state, you_coords, O)
        return fp_state

    def perform_action(self, state, action, player):
        '''
        Write the player's symbol to the tile defined by action.
        '''
        state[action] = player
        return state

    def valid_action(self, state, action):
        '''
        CHeck if the chosen action in a state is valid.
        '''
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

    def get_action_mask(self, state):
        """
        Mask all invalid states. 
        Return a binary list indicating valid moves.
        """
        mask = (state == 0)
        return mask
    
    def mask_actions(self, action_values, action_mask):
        inf_tensor = -torch.inf * torch.ones_like(action_values)
        action_values = torch.where(action_mask == True, action_values, inf_tensor)
        return action_values
    
    def action_to_coords(self, flat_action):
        '''
        Convert a flattened action into a regular action.
        Actions are represented as a single intiger chosen by the agent.
        We need to convert this intiger index into (K, L, x, y) coordinates.
        '''
        i = flat_action
        K = math.floor(i / 9) % 3
        L = math.floor(i / 27)
        x = int(i % 3)
        y = math.floor(i / 3) % 3
        return (K, L, x, y)

    def coords_to_action(self, coords):
        '''
        Convert a (K, L, x, y) action into a flat action.
        The inverse function of action_to_coords.
        '''
        K, L, x, y = coords
        action = K * 27 + L * 9 + x + y * 3
        return action

    def flatten_state(self, state):
        '''
        Flatten 4-dimensional state.
        (3, 3, 3, 3) -> (81)
        '''
        state = torch.transpose(state, 2, 3)
        state = torch.transpose(state, 0, 1)
        state = state.reshape(-1)
        return state
    
    def reshape_state(self, state):
        '''
        Inverse function of flatten_state.
        '''
        state = state.reshape(3, 3, 3, 3)
        state = torch.transpose(state, 1, 0)
        state = torch.transpose(state, 3, 2)
        return state
    
    def evaluate_action(self, state, action):
        '''
        Count the number of points scored by an action.

        Args:
            state: the state following the action
            action: the action taken by the actor

        I have chosen to include the state following the action in order to not rely on
        having to pass or calculate the player.
        '''

        def scoring_position(position, clean_state, action):
            if action in position:
                for coord in position:
                    if clean_state[coord] != 1:
                        return False
                return True
            else:
                return False

        player = state[action]
        clean_state = (state == player)
        score = 0
        for position in self.scoring_positions:
            score += int(scoring_position(position, clean_state, action))
        return score

    def terminal_state(self, state):
        '''
        Returns True if game state is terminal,
        i.e. there are no empty tile left.
        '''
        return (state == E).sum() == 0
    
    def reset(self):
        self.state = self.init_state()
        self.player = X
        self.score = {
            X: 0,
            O: 0
        }
